#include "gtest/gtest.h"

#include <atomic>

#include "DocChangeBucket.h"

using namespace circt::lsp;
using namespace std::chrono;

namespace {
using Bucket = DocChangeBucket<int>;

static std::shared_ptr<Bucket> makeNoDebounceBucket() {
  return std::make_shared<Bucket>();
}

TEST(BucketRegistryTest, GetOrCreateReturnsSameForSameKey) {
  BucketRegistry<Bucket> reg(makeNoDebounceBucket);

  auto a1 = reg.getOrCreate("a");
  auto a2 = reg.getOrCreate("a");
  auto b = reg.getOrCreate("b");

  ASSERT_TRUE(a1);
  ASSERT_TRUE(a2);
  ASSERT_TRUE(b);
  EXPECT_EQ(a1.get(), a2.get());
  EXPECT_NE(a1.get(), b.get());
  EXPECT_EQ(reg.size(), 2u);
}

TEST(BucketRegistryTest, FindAndErase) {
  BucketRegistry<Bucket> reg(makeNoDebounceBucket);

  auto a = reg.getOrCreate("a");
  ASSERT_TRUE(a);
  EXPECT_TRUE(reg.find("a"));

  // Erase calls defaultCancel -> alive=false, and removes from map.
  reg.erase("a");
  EXPECT_FALSE(reg.find("a"));

  // The instance should be canceled.
  EXPECT_FALSE(a->alive.load());
}

TEST(BucketRegistryTest, CancelAll) {
  BucketRegistry<Bucket> reg(makeNoDebounceBucket);
  auto a = reg.getOrCreate("a");
  auto b = reg.getOrCreate("b");

  reg.cancelAll();
  EXPECT_TRUE(reg.empty());
  EXPECT_FALSE(a->alive.load());
  EXPECT_FALSE(b->alive.load());
}
} // namespace
