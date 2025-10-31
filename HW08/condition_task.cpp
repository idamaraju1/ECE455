#include <taskflow/taskflow.hpp>

int main() {
	tf::Executor executor;
	tf::Taskflow taskflow("Condition Task Demo");

	int counter = 0;
	const int limit = 5;

	auto init = taskflow.emplace([&]() {
		std::printf("Initialize counter = %d\n", counter);
	});

	auto loop = taskflow.emplace([&]() {
		std::printf("Loop iteration %d\n", counter);
		counter++;
		return (counter < limit) ? 0 : 1; // 0 => go back, 1 => exit
	}).name("condition");

	auto done = taskflow.emplace([]() {
		std::printf("Loop done.\n");
	});

	init.precede(loop);
	loop.precede(loop, done); // self-edge enables iteration

	executor.run(taskflow).wait();

	return 0;
}