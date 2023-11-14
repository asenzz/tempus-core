#include "include/DaoTestFixture.h"

#include <model/User.hpp>

using svr::datamodel::User;

TEST_F(DaoTestFixture, UserWorkflow)
{
    User_ptr user1 = std::make_shared<User> (
         bigint(), "test_user_name", "test_user@email", "test_user_password", "test_user_another_name", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High );

    ASSERT_FALSE(aci.user_service.exists(user1->get_user_name()));

    aci.user_service.save( user1 );

    User_ptr saved = aci.user_service.get_user_by_user_name(user1->get_user_name());

    ASSERT_EQ(*user1, *saved);

    User_ptr user2 = std::make_shared<User> (
         bigint(), "new_test_user_name", "test_user@new_email", "As3~*\\|/ ", "test_user_another_name", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High );

    aci.user_service.save(user2);

    ASSERT_EQ(1, aci.user_service.remove(user1));

    user1 = aci.user_service.get_user_by_user_name(user2->get_user_name());

    ASSERT_EQ(*user1, *user2);

    aci.user_service.remove(user2);

    ASSERT_FALSE(aci.user_service.exists(user2->get_user_name()));
}
