#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <functional>
#include <future>
#include <vector>
#include <tuple>
#include <memory>
#include <type_traits>


namespace svr {
namespace common {


/******************************************************************************/

struct task_base
{
    virtual void execute() = 0;
    virtual ~task_base() {};
};
using task_base_ptr = std::shared_ptr<task_base>;


struct future_base
{
    virtual void wait() = 0;
    virtual task_base_ptr get_task() const = 0;
    virtual ~future_base(){};
};
using future_base_ptr = std::shared_ptr<future_base>;


namespace detail{


template<int ...> struct index_sequence {};


template<int N, int ...S>
struct index_generator : index_generator<N-1, N-1, S...> {};


template<int ...S>
struct index_generator<0, S...>
{
    typedef index_sequence<S...> type;
};


template<class Result>
struct task_result : public task_base
{
    Result const & get_result() const;
protected:
    Result result;
};


template<typename function, class... function_args>
class task_w_result : public task_result<typename std::result_of<function(function_args...)>::type>
{
    typename std::decay<function>::type func;
    std::tuple<typename std::decay<function_args>::type...> arguments;
    using task_result<typename std::result_of<function(function_args...)>::type>::result;
    template<int ...S> void execute_helper(detail::index_sequence<S...>);
public:
    task_w_result(function f, function_args... args);
    void execute();
    using task_result<typename std::result_of<function(function_args...)>::type>::get_result;
};


template<typename function, class... function_args>
class task_wo_result : public task_base
{
    typename std::decay<function>::type func;
    std::tuple<typename std::decay<function_args>::type...> arguments;
    template<int ...S> void execute_helper(detail::index_sequence<S...>);
public:
    task_wo_result(function f, function_args... args);
    void execute();
};


template<class function_result, bool is_void = std::is_void<function_result>::value>
struct future_helper
{
    future_helper(future_base & fut);
    function_result get();
};


} // namespace detail


template <class Function, class... Args>
inline std::shared_ptr<svr::common::task_base> make_task(Function && func, Args && ... args);


/******************************************************************************/


template<typename future_result>
class future : public future_base, public detail::future_helper<future_result>
{
    future_base_ptr future_impl;
public:
    future(future_base_ptr fut_impl);
    future() : future(nullptr) {};

    using detail::future_helper<future_result>::get;

private:
    void wait();
    task_base_ptr get_task() const ;

};


/******************************************************************************/


class cpu_thread_pool
{
public:
    static cpu_thread_pool & get_instance();
    future_base_ptr execute(task_base_ptr const & task);
    void current_thread_blocked(bool blocked);
    size_t hardware_concurrency() const;
private:
    cpu_thread_pool();
    ~cpu_thread_pool();
    cpu_thread_pool(cpu_thread_pool const &)=delete;
    void operator=(cpu_thread_pool)=delete;

    class Impl;
    Impl & pImpl;
};


template<typename function, class... function_args>
future<typename std::result_of<function(function_args...)>::type> async( function&& f, function_args&&... args );


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


// Sets waiting state for the current thread
struct current_thread_wstate_guard
{
    current_thread_wstate_guard();
    ~current_thread_wstate_guard();
};


namespace detail{


template<bool is_void>
struct task_result_helper
{
    template <typename function, typename... function_args>
    static std::shared_ptr<svr::common::task_base> make_task(function && func, function_args&& ... args);
};


template<>
struct task_result_helper<false>
{
    template <class Function, class... Args>
    static std::shared_ptr<svr::common::task_base> make_task(Function && func, Args&&  ... args)
    {
        return std::shared_ptr<svr::common::task_base>(new task_w_result<Function, Args...>(std::forward<Function>(func), std::forward<Args>(args)...));
    }
};


template<>
struct task_result_helper<true>
{
    template <class Function, class... Args>
    static std::shared_ptr<svr::common::task_base> make_task(Function && func, Args && ... args)
    {
        return std::shared_ptr<svr::common::task_base>(new task_wo_result<Function, Args...>(std::forward<Function>(func), std::forward<Args>(args)...));
    }
};


template<class Result>
Result const & task_result<Result>::get_result() const
{
    return result;
}


template<typename function, typename... function_args>
task_w_result<function, function_args...>::task_w_result(function f, function_args... args)
: func(f)
, arguments(args...)
{}


template<typename function, typename... Args>
void task_w_result<function, Args...>::execute()
{
    execute_helper(typename detail::index_generator<sizeof...(Args)>::type());
}


template<typename function, typename... Args>
template<int ...S>
void task_w_result<function, Args...>::execute_helper(detail::index_sequence<S...>)
{
    auto fn = std::bind(func, std::get<S>(arguments) ...);
    result = fn();

//    result = func(std::get<S>(arguments)...);
}


template<typename function, typename... Args>
task_wo_result<function, Args...>::task_wo_result(function f, Args... args)
: func(f)
, arguments(args...)
{}


template< class Function, class... Args>
void task_wo_result<Function, Args...>::execute()
{
    execute_helper(typename detail::index_generator<sizeof...(Args)>::type());
}


template< class Function, class... Args>
template<int ...S>
void task_wo_result<Function, Args...>::execute_helper(detail::index_sequence<S...>)
{
    auto fn = std::bind(func, std::get<S>(arguments) ...);
    fn();
}


template<>
struct future_helper<void, true>
{
    future_helper(future_base * fut)
    : fut(fut)
    {}

    void get()
    {
        current_thread_wstate_guard scoped;
        fut->wait();
    };

private:
    future_base * fut;
};


template<class Result>
struct future_helper<Result, false>
{
    future_helper(future_base * fut)
    : fut(fut)
    {}

    Result get()
    {
        {
            current_thread_wstate_guard scoped;
            fut->wait();
        }
        return static_cast<task_result<Result>*>(fut->get_task().get())->get_result();
    }

private:
    future_base * fut;
};


} // namespace detail


/******************************************************************************/

template<typename Result>
future<Result>::future(future_base_ptr fut_impl)
: detail::future_helper<Result>(static_cast<future_base*>(fut_impl.get()))
, future_impl(fut_impl)
{
}


template<typename Result>
void future<Result>::wait()
{
    future_impl->wait();
}


template<typename Result>
task_base_ptr future<Result>::get_task() const
{
    return future_impl->get_task();
}


/******************************************************************************/

template <class Function, class... Args>
inline task_base_ptr make_task(Function && func, Args && ... args)
{
    return detail::task_result_helper<std::is_void<typename std::result_of<Function(Args...)>::type>::value>
            ::make_task(std::forward<Function>(func), std::forward<Args>(args)...);
}


/******************************************************************************/


template< class Function, class... Args>
future<typename std::result_of<Function(Args...)>::type> async( Function&& f, Args&&... args )
{
    return cpu_thread_pool::get_instance().execute(make_task(std::forward<Function>(f), std::forward<Args>(args)...));
}


/******************************************************************************/

}

using common::async;
using common::future;

//using std::async;
//using std::future;

}

#endif /* THREADPOOL_HPP */

