#include <iostream>

#include <pqxx/pqxx>
#include <boost/program_options.hpp>

struct Order
{
    double px, size, stoploss, trailingstop, takeprofit;
    static Order mapRow(const pqxx::row& rowSet)
    {
        return Order({
           rowSet["px"].as<double>(0),
           rowSet["size"].as<double>(0),
           rowSet["stoploss"].as<double>(0),
           rowSet["trailingstop"].as<double>(0),
           rowSet["takeprofit"].as<double>(0)
        });
    }
};

bool order_px_less_px(double px, Order const & ord)
{
    return px < ord.px;
}

bool order_px_less(Order const & lhs, Order const & rhs)
{
    return lhs.px < rhs.px;
}

std::basic_ostream<char> & operator<<(std::basic_ostream<char> & ostr, Order const & order)
{
    ostr << "px: " << order.px << " size: " << order.size << " stoploss: "
            << order.stoploss << " trailingstop: " << order.trailingstop << " takeprofit: " << order.takeprofit;
    return ostr;
}

struct OrderRequest
{
    bool is_buy;
    double px, delta, length;
};

OrderRequest parse(int argc, char** argv);

int main(int argc, char** argv)
{
    OrderRequest o_r = parse(argc, argv);

    pqxx::connection conn("dbname=svrwave user=svrwave password=svrwave host=localhost port=5432");
    conn.set_session_var("search_path", "svr,victor");

    std::vector<Order> order_templates;

    {
        pqxx::work trx(conn);

        pqxx::result result = trx.exec("select px, size, stoploss, trailingstop, takeprofit from order_template order by px");

        for(const auto& row : result)
            order_templates.push_back(Order::mapRow(row));
    }

    if(order_templates.empty())
    {
        std::cerr << "Cannot load order templates." << std::endl;
        exit(1);
    }

    double const px_mod = o_r.px - long(o_r.px) + long(o_r.px) % 100
        , px_addition =  o_r.px - px_mod
        , granularity = 100
        ;

    std::vector<Order> result;

    if(!o_r.is_buy)
    {
        auto iter = std::upper_bound(order_templates.begin(), order_templates.end(), px_mod, order_px_less_px);

        if(iter == order_templates.end()) iter = order_templates.end() -1;

        double acc_move = 0;
        size_t rounds = 0;

        while(true)
        {
            bool take_this_order = std::abs(iter->px - px_mod) > o_r.delta;
            if( take_this_order )
                result.push_back(Order({iter->px + px_addition - rounds * granularity, -iter->size, iter->stoploss, iter->trailingstop, iter->takeprofit}));

            double prevpx = iter->px;

            if(iter == order_templates.begin())
            {
                iter = order_templates.end();
                ++rounds;
            }
            --iter;

            if(take_this_order)
            {
                if(iter->px > prevpx)
                    prevpx += granularity;
                acc_move += prevpx - iter->px;
            }

            if(acc_move >= o_r.length)
                break;
        }
    }
    else
    {
        auto iter = std::upper_bound(order_templates.begin(), order_templates.end(), px_mod, order_px_less_px);

        if(iter == order_templates.end()) iter = order_templates.begin();

        double acc_move = 0;
        size_t rounds = 0;

        while(true)
        {
            bool take_this_order = std::abs(iter->px - px_mod) > o_r.delta;
            if( take_this_order )
            {
                result.push_back(Order({iter->px + px_addition + rounds * granularity, iter->size, iter->stoploss, iter->trailingstop, iter->takeprofit}));
            }

            double prevpx = iter->px;

            ++iter;

            if(iter == order_templates.end())
            {
                iter = order_templates.begin();
                ++rounds;
            }

            if(take_this_order)
            {
                if(iter->px < prevpx)
                    prevpx -= granularity;
                acc_move +=  iter->px - prevpx;
            }

            if(acc_move >= o_r.length)
                break;
        }
    }


    std::sort(result.begin(), result.end(), order_px_less);

    for(auto ord : result)
    {
        std::cout << ord << std::endl;
    }

}

OrderRequest parse(int argc, char** argv)
{
    boost::program_options::options_description gen_desc = boost::program_options::options_description("Daemon options");
    gen_desc.add_options()
        ("help",     "produce help message")
        ("side,s",   boost::program_options::value<std::string>()->required(), "order side. Available sides are b and s")
        ("px,p",     boost::program_options::value<double>()->required(), "curent price")
        ("delta,d",  boost::program_options::value<double>()->required(), "order price delta")
        ("length,l", boost::program_options::value<double>()->required(), "order price length")
    ;

    boost::program_options::variables_map vm;

    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(gen_desc).run(), vm);

    if (vm.count("help"))
    {
        std::cout << gen_desc << std::endl;
        exit(1);
    }

    boost::program_options::notify(vm);

    if(vm["side"].as<std::string>() != "b" && vm["side"].as<std::string>() != "s")
    {
        std::cout << "Available sides are b and s" << std::endl;
        exit(1);
    }

    return OrderRequest({vm["side"].as<std::string>() == "b", vm["px"].as<double>(), vm["delta"].as<double>(), vm["length"].as<double>()});

}

