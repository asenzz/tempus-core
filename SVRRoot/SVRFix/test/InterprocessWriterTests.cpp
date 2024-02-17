#include <gtest/gtest.h>

#include "../src/InterprocessWriter.hpp"
#include "InterprocessReader.hpp"

TEST(InterprocessWriterTests, TestWrite)
{
    size_t const writer_buffer_size = 1000;
    constexpr auto mm_file_name = "InterprocessWriterTests_mm_file";

    svr::fix::mm_file_writer writer(mm_file_name, writer_buffer_size);
    std::vector<svr::fix::bid_ask_spread> etalon_bas; etalon_bas.reserve(2*1000);

    for(size_t i = 0; i < writer_buffer_size; ++i)
    {
        etalon_bas.emplace_back(1.1 * i, 2 * i, 3.1 * i, 4 * i, bpt::microsec_clock::local_time());
        writer.write(etalon_bas.back());
    }

    svr::fix::mm_file_reader reader(mm_file_name);

    auto all1 = reader.read_all();

    EXPECT_TRUE(all1.size() == writer_buffer_size);

    EXPECT_TRUE(all1 == etalon_bas);

    auto new1 = reader.read_new(all1.back().time);

    EXPECT_TRUE(new1.size() == 0UL);

    for(size_t i = writer_buffer_size; i <  3 * writer_buffer_size / 2; ++i)
    {
        etalon_bas.emplace_back(1.1 * i, 2 * i, 3.1 * i, 4 * i, bpt::microsec_clock::local_time());
        writer.write(etalon_bas.back());
    }

    auto all2 = reader.read_all();

    EXPECT_TRUE(all2.size() == writer_buffer_size);

    auto iter = etalon_bas.begin() + writer_buffer_size / 2;

    for(auto bas : all2)
        EXPECT_TRUE(bas == *iter++);

    auto new2 = reader.read_new(all1.back().time);

    EXPECT_TRUE(new2.size() == writer_buffer_size / 2);

    iter = etalon_bas.begin() + writer_buffer_size;

    for(auto bas : new2)
        EXPECT_TRUE(bas == *iter++);

    svr::fix::bid_ask_spread minmax (
          std::numeric_limits<double>::min()
        , std::numeric_limits<size_t>::min()
        , std::numeric_limits<double>::max()
        , std::numeric_limits<size_t>::max()
        , bpt::microsec_clock::local_time()
    );

    writer.write(minmax);

    auto new3 = reader.read_new(new2.back().time);

    EXPECT_TRUE(new3.size() == 1UL);

    EXPECT_TRUE(new3.front() == minmax);
}


