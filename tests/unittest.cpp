// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#include "gtest/gtest.h"
#include "ibcomm/util.h"

namespace {

class IBCommUtilTest : public ::testing::Test {
 protected:
};

TEST_F(IBCommUtilTest, ParseNumberZero) {
  EXPECT_EQ(0, util::parse_number("0"));
  EXPECT_EQ(0, util::parse_number("0b"));
  EXPECT_EQ(0, util::parse_number("0B"));
  EXPECT_EQ(0, util::parse_number("0k"));
  EXPECT_EQ(0, util::parse_number("0kb"));
  EXPECT_EQ(0, util::parse_number("0K"));
  EXPECT_EQ(0, util::parse_number("0m"));
  EXPECT_EQ(0, util::parse_number("0mb"));
  EXPECT_EQ(0, util::parse_number("0M"));
  EXPECT_EQ(0, util::parse_number("0g"));
  EXPECT_EQ(0, util::parse_number("0gb"));
  EXPECT_EQ(0, util::parse_number("0G"));

  EXPECT_EQ(0, util::parse_number("-0"));
  EXPECT_EQ(0, util::parse_number("-0b"));
  EXPECT_EQ(0, util::parse_number("-0B"));
  EXPECT_EQ(0, util::parse_number("-0k"));
  EXPECT_EQ(0, util::parse_number("-0kb"));
  EXPECT_EQ(0, util::parse_number("-0K"));
  EXPECT_EQ(0, util::parse_number("-0m"));
  EXPECT_EQ(0, util::parse_number("-0mb"));
  EXPECT_EQ(0, util::parse_number("-0M"));
  EXPECT_EQ(0, util::parse_number("-0g"));
  EXPECT_EQ(0, util::parse_number("-0gb"));
  EXPECT_EQ(0, util::parse_number("-0G"));
}

TEST_F(IBCommUtilTest, ParseNumberPositive) {
  EXPECT_EQ(1, util::parse_number("1"));
  EXPECT_EQ(1, util::parse_number("1b"));
  EXPECT_EQ(1, util::parse_number("1B"));

  EXPECT_EQ(1024, util::parse_number("1k"));
  EXPECT_EQ(1024, util::parse_number("1kb"));

  EXPECT_EQ(31 * 1024, util::parse_number("31k"));
  EXPECT_EQ(31 * 1024, util::parse_number("31kb"));

  EXPECT_EQ(713ul * 1024 * 1024, util::parse_number("713m"));
  EXPECT_EQ(713ul * 1024 * 1024, util::parse_number("713mb"));
}

TEST_F(IBCommUtilTest, ParseNumberMalformed) {
  ASSERT_THROW(util::parse_number("0.5"), util::MalformedNumber);
  ASSERT_THROW(util::parse_number("a"), util::MalformedNumber);
  ASSERT_THROW(util::parse_number("b"), util::MalformedNumber);
  ASSERT_THROW(util::parse_number("B"), util::MalformedNumber);
  ASSERT_THROW(util::parse_number("0x"), util::MalformedNumber);
  ASSERT_THROW(util::parse_number("97MiB"),
               util::MalformedNumber);  // "Mibi byte" is not supported
}

TEST_F(IBCommUtilTest, get_exp_of_two) {
  ASSERT_EQ(0, util::get_exp_of_two(0));
  ASSERT_EQ(1, util::get_exp_of_two(2));
  ASSERT_EQ(7, util::get_exp_of_two(128));
  ASSERT_EQ(0, util::get_exp_of_two(127));
  ASSERT_EQ(0, util::get_exp_of_two(129));
}

}  // namespace
