#include <InterprocessReader.hpp>

#include <boost/interprocess/managed_shared_memory.hpp>

#include "common/logging.hpp"

namespace svr {
namespace fix {

/******************************************************************************/
/*                             Deserialization                                */
/******************************************************************************/

template <class T>
class interprocess_serializer_reader
{
public:
    static constexpr size_t size = 0;
    inline void  read(char const * buffer, T & what);
};

/******************************************************************************/

template<>
class interprocess_serializer_reader<bid_ask_spread>
{
public:
    static constexpr size_t double_size = 8;
    static constexpr size_t size_t_size = 8;

    static constexpr size_t size = 2 * double_size + 3 * size_t_size;

    static inline void read(char const * buffer, bid_ask_spread & value)
    {
        value.bid_px = *reinterpret_cast<double const *>(buffer) ;
        buffer += double_size;
        value.bid_qty = *reinterpret_cast<size_t const *>(buffer);
        buffer += size_t_size;

        value.ask_px = *reinterpret_cast<double const *>(buffer);
        buffer += double_size;
        value.ask_qty = *reinterpret_cast<size_t const *>(buffer);
        buffer += size_t_size;

        static const bpt::ptime epoch = bpt::from_iso_string("20100101T000000");

        value.time = epoch + bpt::microseconds(*reinterpret_cast<size_t const *>(buffer));
    }

private:
    static_assert(sizeof(double) == double_size, "Data type size has changed. Please fix the size and perform full rebuild");
    static_assert(sizeof(size_t) == size_t_size, "Data type size has changed. Please fix the size and perform full rebuild");
};

typedef interprocess_serializer_reader<bid_ask_spread> bid_ask_spread_reader;

/******************************************************************************/

template<>
class interprocess_serializer_reader<mm_file_reader::bid_ask_spread_container>
{
public:
    static constexpr size_t double_size = 8;
    static constexpr size_t size_t_size = 8;

    static constexpr size_t size = 2 * double_size + 3 * size_t_size;

    static void read(char const * buffer, mm_file_reader::bid_ask_spread_container & value)
    {
        const size_t sz = *reinterpret_cast<size_t const *>(buffer) ;
        buffer += size_t_size;

        value.reserve(sz);

        for(DTYPE(sz) i = 0; i < sz; ++i)
        {
            static bid_ask_spread_reader serializer;
            bid_ask_spread tmp;
            serializer.read(buffer, tmp);
            value.emplace_back(tmp);
            buffer += serializer.size;
        }
    }


    void read(char const * buffer, mm_file_reader::bid_ask_spread_container & value, bpt::ptime const & after)
    {
        static bid_ask_spread_reader serializer;

        size_t const sz = *reinterpret_cast<size_t const *>(buffer) ;
        buffer += size_t_size + (sz - 1) * serializer.size;

        std::deque<bid_ask_spread> tmp_queue;

        for(long i = long(sz-1); i >= 0; --i)
        {
            bid_ask_spread tmp_bas;
            serializer.read(buffer, tmp_bas);
            if(tmp_bas.time > after)
                tmp_queue.emplace_front(tmp_bas);
            else
                break;
            buffer -= serializer.size;
        }

        value.reserve(tmp_queue.size());
        std::copy(tmp_queue.begin(), tmp_queue.end(), std::back_inserter(value));
    }


private:
    static_assert(sizeof(double) == double_size, "Data type size has changed. Please fix the size and perform full rebuild");
    static_assert(sizeof(size_t) == size_t_size, "Data type size has changed. Please fix the size and perform full rebuild");
};

typedef interprocess_serializer_reader<mm_file_reader::bid_ask_spread_container> bid_ask_spread_file_reader;


/******************************************************************************/
/*                             Implementation                                 */
/******************************************************************************/

struct mm_file_reader::Impl
{
    boost::interprocess::managed_shared_memory mm_file;
    char const * mm_file_buffer;

    explicit Impl(std::string const & mm_file_name) : mm_file (boost::interprocess::managed_shared_memory(boost::interprocess::open_only, mm_file_name.c_str()))
    {
        const auto [fst, snd] = mm_file.find<boost::interprocess::managed_shared_memory::handle_t>("mm_file_handle");
        if (snd != 1) LOG4_THROW("mm_file_reader error: cannot find mm file buffer handle");
        mm_file_buffer = static_cast<char const *>(mm_file.get_address_from_handle(*fst));
    }

    mm_file_reader::bid_ask_spread_container read_all() const
    {
        mm_file_reader::bid_ask_spread_container result;
        bid_ask_spread_file_reader reader;
        reader.read(mm_file_buffer, result);
        return result;
    }

    mm_file_reader::bid_ask_spread_container read_new(bpt::ptime const & last_time) const
    {
        mm_file_reader::bid_ask_spread_container result;
        bid_ask_spread_file_reader reader;
        reader.read(mm_file_buffer, result, last_time);
        return result;
    }

};


mm_file_reader::mm_file_reader(std::string const & mm_file_name) : pImpl(* new Impl(mm_file_name))
{}


mm_file_reader::~mm_file_reader()
{
    delete & pImpl;
}


mm_file_reader::bid_ask_spread_container mm_file_reader::read_all() const
{
    return pImpl.read_all();
}


mm_file_reader::bid_ask_spread_container mm_file_reader::read_new(bpt::ptime const & last_time) const
{
    return pImpl.read_new(last_time);
}

}}

