/*
 * main.cpp
 *
 *  Created on: Jul 29, 2014
 *      Author: vg
 *//*


#include "config/common.hpp"
#include "UserService.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/UserDAO.hpp"
#include "UserService.hpp"
using namespace svr::dao;
using namespace svr::business;
using namespace std;

int main(){

	auto svc = make_shared<UserService>(make_shared<UserDAO>(make_shared<DataSource>("dbname=svrtest user=svradmin password=svrsvrsvr host=localhost")));
	cout << svc->get_user_by_id(42).to_string() << endl;
	return 0;
}
*/
