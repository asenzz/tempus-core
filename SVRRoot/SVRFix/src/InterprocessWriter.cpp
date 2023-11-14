#include "InterprocessWriter.hpp"

#include <boost/circular_buffer.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace svr {
namespace fix {

void interprocess_writer_base::write(bid_ask_spread const & spread)
{
    do_write(spread);
}


/******************************************************************************/
/*                             Serialization                                  */
/******************************************************************************/

template<class T>
class interprocess_serializer_writer
{
    inline void write(T const & value, char * buffer);
};


template<>
class interprocess_serializer_writer<bid_ask_spread>
{
public:
    static size_t const double_size = 8;
    static size_t const size_t_size = 8;

    static size_t const size = 2 * double_size + 3 * size_t_size;
    void write(bid_ask_spread const & value, char * buffer)
    {
        *reinterpret_cast<double*>(buffer) = value.bid_px;
        buffer += double_size;
        *reinterpret_cast<size_t*>(buffer) = value.bid_qty;
        buffer += size_t_size;

        *reinterpret_cast<double*>(buffer) = value.ask_px;
        buffer += double_size;
        *reinterpret_cast<size_t*>(buffer) = value.ask_qty;
        buffer += size_t_size;

        static const bpt::ptime epoch = bpt::from_iso_string("20100101T000000");

        size_t tm = (value.time - epoch).total_microseconds();
        *reinterpret_cast<size_t*>(buffer) = tm;
    }

private:
    static_assert(sizeof(double) == double_size, "Data type size has changed. Please fix the size and perform full rebuild");
    static_assert(sizeof(size_t) == size_t_size, "Data type size has changed. Please fix the size and perform full rebuild");
};

typedef interprocess_serializer_writer<bid_ask_spread> bid_ask_spread_writer;

/******************************************************************************/

typedef boost::circular_buffer<bid_ask_spread> bid_ask_spread_circular_file;

template<>
class interprocess_serializer_writer<bid_ask_spread_circular_file>
{
public:
    static size_t const size_t_size = 8;
    static size_t const size = size_t_size;

    void write(bid_ask_spread_circular_file const & file, char * buffer)
    {
        size_t const sz = file.size();
        *reinterpret_cast<size_t*>(buffer) = sz;
        buffer += size_t_size;

        for(auto const bas : file)
        {
            static bid_ask_spread_writer serializer;
            serializer.write(bas, buffer);
            buffer += serializer.size;
        }
    }

    constexpr size_t get_size(size_t no_of_elements)
    {
        return size_t_size + no_of_elements * bid_ask_spread_writer::size;
    }

private:
    static_assert(sizeof(size_t) == size_t_size, "Data type size has changed. Please fix the size and perform full rebuild");
};

typedef interprocess_serializer_writer<bid_ask_spread_circular_file> bid_ask_spread_circular_file_serializer;

/******************************************************************************/

struct mm_file_writer::Impl
{
    boost::interprocess::managed_shared_memory msm;
    std::string const mm_file_name;
    size_t no_of_elements;
    bid_ask_spread_circular_file raw_data_file;
    boost::interprocess::managed_shared_memory * mm_file;
    char * mm_file_buffer;
    bid_ask_spread_circular_file_serializer serializer;


    Impl(std::string const & mm_file_name, size_t no_of_elements)
    : mm_file_name(mm_file_name)
    , no_of_elements(no_of_elements)
    , raw_data_file(no_of_elements)
    , mm_file(nullptr)
    {
        cleanup();
        size_t file_size = serializer.get_size(no_of_elements);
        mm_file = new boost::interprocess::managed_shared_memory(boost::interprocess::create_only, mm_file_name.c_str(), file_size + 1024);

        boost::interprocess::managed_shared_memory::handle_t * p_mm_file_handle
                = mm_file->construct<boost::interprocess::managed_shared_memory::handle_t>("mm_file_handle")(0);

        mm_file_buffer = reinterpret_cast<char *>(mm_file->allocate(file_size));

        *p_mm_file_handle = mm_file->get_handle_from_address(mm_file_buffer);
    }

    ~Impl()
    {
        mm_file->deallocate(mm_file_buffer);
        delete mm_file;
        cleanup();
    }

    void cleanup()
    {
        boost::interprocess::shared_memory_object::remove(mm_file_name.c_str());
    }

    void write(bid_ask_spread const & spread)
    {
        raw_data_file.push_back(spread);
        serializer.write(raw_data_file, mm_file_buffer);
    }
};

/******************************************************************************/

mm_file_writer::mm_file_writer(std::string const & mm_file_name, size_t no_of_elements)
: pImpl( * new Impl(mm_file_name, no_of_elements) )
{
}


mm_file_writer::~mm_file_writer()
{
    delete & pImpl;
}


void mm_file_writer::do_write(bid_ask_spread const & spread)
{
    pImpl.write(spread);
}


}}
