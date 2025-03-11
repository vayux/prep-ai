# Concurrency

## [**Blocking Queue | Bounded Buffer | Consumer Producer**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/blocking-queue--bounded-buffer--consumer-producer)

A blocking queue is defined as a queue which blocks the caller of the enqueue method if there's no more capacity to add the new item being enqueued. Similarly, the queue blocks the dequeue caller if there are no items in the queue. Also, the queue notifies a blocked enqueuing thread when space becomes available and a blocked dequeuing thread when an item becomes available in the queue.

[](https://www.educative.io/api/collection/5307417243942912/5668546535227392/page/5222874270924800/image/6027491695132672?page_type=collection_lesson)

```python
from threading import Thread
from threading import Condition
from threading import current_thread
import time
import random

class BlockingQueue:

    def __init__(self, max_size):
        self.max_size = max_size
        self.curr_size = 0
        self.cond = Condition()
        self.q = []

    def dequeue(self):

        self.cond.acquire()
        while self.curr_size == 0:
            self.cond.wait()

        item = self.q.pop(0)
        self.curr_size -= 1

        self.cond.notifyAll()
        self.cond.release()

        return item

    def enqueue(self, item):

        self.cond.acquire()
        while self.curr_size == self.max_size:
            self.cond.wait()

        self.q.append(item)
        self.curr_size += 1

        self.cond.notifyAll()
        print("\ncurrent size of queue {0}".format(self.curr_size), flush=True)
        self.cond.release()

def consumer_thread(q):
    while 1:
        item = q.dequeue()
        print("\n{0} consumed item {1}".format(current_thread().getName(), item), flush=True)
        time.sleep(random.randint(1, 3))

def producer_thread(q, val):
    item = val
    while 1:
        q.enqueue(item)
        item += 1
        time.sleep(0.1)

if __name__ == "__main__":
    blocking_q = BlockingQueue(5)

    consumerThread1 = Thread(target=consumer_thread, name="consumer-1", args=(blocking_q,), daemon=True)
    consumerThread2 = Thread(target=consumer_thread, name="consumer-2", args=(blocking_q,), daemon=True)
    producerThread1 = Thread(target=producer_thread, name="producer-1", args=(blocking_q, 1), daemon=True)
    producerThread2 = Thread(target=producer_thread, name="producer-2", args=(blocking_q, 100), daemon=True)

    consumerThread1.start()
    consumerThread2.start()
    producerThread1.start()
    producerThread2.start()

    time.sleep(15)
    print("Main thread exiting")
```

# [**Non-Blocking Queue**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/non-blocking-queue)

This lesson is a follow-up to the blocking queue question and explores the various ways in which we can make a blocking queue non-blocking.

## **Non-Blocking Queue**

### **Problem**

We have seen the blocking version of a queue in the previous question that blocks a producer or a consumer when the queue is full or empty respectively. In this problem, you are asked to implement a queue that is non-blocking. The requirement on non-blocking is intentionally left open to interpretation to see how you think through the problem.

This question is inspired by one of [David Beazley's](https://www.dabeaz.com/) Python talks.

## **Solution**

Let's first define the notion of **non-blocking**. If a consumer or a producer can successfully enqueue or dequeue an item, it is considered non-blocking. However, if the queue is full or empty then a producer or a consumer (respectively) need not wait until the queue can be added to or taken from.

### 

```python
from threading import Thread
from threading import Lock
from threading import RLock
from threading import current_thread
from concurrent.futures import Future
import time
import random

class NonBlockingQueue:

    def __init__(self, max_size):
        self.max_size = max_size
        self.q = []
        self.q_waiting_puts = []
        self.q_waiting_gets = []
        self.lock = RLock()
        #self.lock = Lock()

    def dequeue(self):

        result = None
        future = None

        with self.lock:
            curr_size = len(self.q)

            if curr_size != 0:
                result = self.q.pop(0)

                # remember to resolve a pending future for
                # a put request
                if len(self.q_waiting_puts) != 0:
                    self.q_waiting_puts.pop(0).set_result(True)

            else:
                # queue is empty so create a future for a get
                # request
                future = Future()
                self.q_waiting_gets.append(future)

        return result, future

    def enqueue(self, item):

        # print("size {0}".format(len(self.q_waiting_puts)))

        future = None
        with self.lock:
            curr_size = len(self.q)

            # queue is full so create a future for a put
            # request
            if curr_size == self.max_size:
                future = Future()
                self.q_waiting_puts.append(future)

            else:
                self.q.append(item)

                # remember to resolve a pending future for
                # a get request
                if len(self.q_waiting_gets) != 0:
                    future_get = self.q_waiting_gets.pop(0)
                    future_get.set_result(True)

        return future

def consumer_thread(q):
    while 1:
        item, future = q.dequeue()

        if item is None:
            print("\nConsumer received a future but we are ignoring it")
        else:
            print("\n{0} consumed item {1}".format(current_thread().getName(), item), flush=True)

        # slow down the consumer
        time.sleep(random.randint(1, 3))

def retry_enqueue(future):
    print("\nCallback invoked by thread {0}".format(current_thread().getName()))
    item = future.item
    q = future.q
    new_future = q.enqueue(item)

    if new_future is not None:
        new_future.item = item
        new_future.q = q
        new_future.add_done_callback(retry_enqueue)
    else:
        print("\n{0} successfully added on a retry".format(item))

def producer_thread(q):
    item = 1
    while 1:
        future = q.enqueue(item)
        if future is not None:
            future.item = item
            future.q = q
            future.add_done_callback(retry_enqueue)

        item += 1

        # slow down the producer
        time.sleep(0.1)

if __name__ == "__main__":
    no_block_q = NonBlockingQueue(5)

    consumerThread1 = Thread(target=consumer_thread, name="consumer", args=(no_block_q,), daemon=True)
    producerThread1 = Thread(target=producer_thread, name="producer", args=(no_block_q,), daemon=True)

    consumerThread1.start()
    producerThread1.start()

    time.sleep(15)
    print("\nMain thread exiting")
```

# [**Rate Limiting Using Token Bucket Filter**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/rate-limiting-using-token-bucket-filter)

Implementing rate limiting using a naive token bucket filter algorithm.

## **Rate Limiting Using Token Bucket Filter**

*This is an actual interview question asked at Uber and Oracle.*

Imagine you have a bucket that gets filled with tokens at the rate of 1 token per second. The bucket can hold a maximum of N tokens. Implement a thread-safe class that lets threads get a token when one is available. If no token is available, then the token-requesting threads should block.

The class should expose an API called **`get_token()`** that various threads can call to get a token.

```python
from threading import Thread
from threading import current_thread
from threading import Semaphore
from threading import current_thread
from threading import Lock
from threading import Barrier
import random
import time

class TokenBucketFilter:

    def __init__(self, MAX_TOKENS):
        self.MAX_TOKENS = MAX_TOKENS
        self.last_request_time = time.time()
        self.possible_tokens = 0
        self.lock = Lock()

    def get_token(self):

        with self.lock:
            self.possible_tokens += int((time.time() - self.last_request_time))

            if self.possible_tokens > self.MAX_TOKENS:
                self.possible_tokens = self.MAX_TOKENS

            if self.possible_tokens == 0:
                time.sleep(1)
            else:
                self.possible_tokens -= 1

            self.last_request_time = time.time()

            print("Granting {0} token at {1} ".format(current_thread().getName(), int(time.time())));

if __name__ == "__main__":

    token_bucket_filter = TokenBucketFilter(5)

    time.sleep(10)

    threads = list()
    for _ in range(0, 12):
        threads.append(Thread(target=token_bucket_filter.get_token))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
```

```python
from threading import Thread
from threading import Condition
from threading import current_thread
import time

class TokenBucketFilterFactory:

    @staticmethod
    def makeTokenBucketFilter(capacity):
        tbf = MultithreadedTokenBucketFilter(capacity)
        tbf.initialize();
        return tbf;

class MultithreadedTokenBucketFilter:
    def __init__(self, maxTokens):
        self.MAX_TOKENS = int(maxTokens)
        self.possibleTokens  = int(0)
        self.ONE_SECOND = int(1)
        self.cond = Condition()

    def initialize(self):
        dt = Thread(target = self.daemonThread);
        dt.setDaemon(True);
        dt.start();
                
    def daemonThread(self):
        while True:
            self.cond.acquire()
            if self.possibleTokens < self.MAX_TOKENS:
                self.possibleTokens = self.possibleTokens + 1;
            self.cond.notify() 
            self.cond.release()
            
            time.sleep(self.ONE_SECOND);
    
    def getToken(self):
        self.cond.acquire()
        while self.possibleTokens == 0:
            self.cond.wait()
        self.possibleTokens = self.possibleTokens - 1
        self.cond.release()

        print("Granting " + current_thread().getName() + " token at " + str(time.time()))
    
```

## [**Thread Safe Deferred Callback**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/thread-safe-deferred-callback)

Design and implement a thread-safe class that allows registration of callback methods that are executed after a user specified time interval in seconds has elapsed.

## **Solution**

Let us try to understand the problem without thinking about concurrency. Let's say our class exposes an API called **`add_action()`** that'll take a parameter **`action`**, which will get executed after user specified seconds. Anyone calling this API should be able to specify after how many seconds should our class invoke the passed-in action.

```python
from threading import Condition
from threading import Thread
import heapq
import time
import math

class DeferredCallbackExecutor():
    def __init__(self):
        self.actions = list()
        self.cond = Condition()
        self.sleep = 0

    def add_action(self, action):
        # add exec_at time for the action
        action.execute_at = time.time() + action.exec_secs_after

        self.cond.acquire()
        heapq.heappush(self.actions, action)
        self.cond.notify()
        self.cond.release()

    def start(self):

        while True:
            self.cond.acquire()

            while len(self.actions) is 0:
                self.cond.wait()

            while len(self.actions) is not 0:

                # calculate sleep duration
                next_action = self.actions[0]
                sleep_for = next_action.execute_at - math.floor(time.time())
                if sleep_for <= 0:
                    # time to execute action
                    break

                self.cond.wait(timeout=sleep_for)

            action_to_execute_now = heapq.heappop(self.actions)
            action_to_execute_now.action(*(action_to_execute_now,))

            self.cond.release()

class DeferredAction(object):
    def __init__(self, exec_secs_after, name, action):
        self.exec_secs_after = exec_secs_after
        self.action = action
        self.name = name

    def __lt__(self, other):
        return self.execute_at < other.execute_at

def say_hi(action):
        print("hi, I am {0} executed at {1} and required at {2}".format(action.name, math.floor(time.time()),
                                                                    math.floor(action.execute_at)))

if __name__ == "__main__":
    action1 = DeferredAction(3, ("A",), say_hi)
    action2 = DeferredAction(2, ("B",), say_hi)
    action3 = DeferredAction(1, ("C",), say_hi)
    action4 = DeferredAction(7, ("D",), say_hi)

    executor = DeferredCallbackExecutor()
    t = Thread(target=executor.start, daemon=True)
    t.start()

    executor.add_action(action1)
    executor.add_action(action2)
    executor.add_action(action3)
    executor.add_action(action4)

    # wait for all actions to execute
    time.sleep(15)
```

# [**Implementing Semaphore**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/implementing-semaphore)

Learn how to design and implement a simple semaphore class in Python.

## **Implementing Semaphore**

Python does provide its own implementation of **`Semaphore`** and **`BoundedSemaphore`**, however, we want to implement a semaphore with a slight twist.

Briefly, a semaphore is a construct that allows some threads to access a fixed set of resources in parallel. Always think of a semaphore as having a fixed number of permits to give out. Once all the permits are given out, requesting threads, need to wait for a permit to be returned before proceeding forward.

Your task is to implement a semaphore which takes in its constructor the maximum number of permits allowed and is also initialized with the same number of permits. Additionally, if all the permits have been given out, the semaphore blocks threads attempting to acquire it.

```python
from threading import Condition
from threading import Thread
import time

class CountSemaphore():

    def __init__(self, permits):
        self.max_permits = permits
        self.given_out = 0
        self.cond_var = Condition()

    def acquire(self):
        self.cond_var.acquire()
        while self.given_out == self.max_permits:
            self.cond_var.wait()

        self.given_out += 1
        self.cond_var.notifyAll()
        self.cond_var.release()

    def release(self):

        self.cond_var.acquire()

        while self.given_out == 0:
            self.cond_var.wait()

        self.given_out -= 1
        self.cond_var.notifyAll()
        self.cond_var.release()

def task1(sem):
    # consume the first permit
    sem.acquire()

    print("acquiring")
    sem.acquire()

    print("acquiring")
    sem.acquire()

    print("acquiring")
    sem.acquire()

def task2(sem):
    time.sleep(2)
    print("releasing")
    sem.release()

    time.sleep(2)
    print("releasing")
    sem.release()

    time.sleep(2)
    print("releasing")
    sem.release()

if __name__ == "__main__":
    sem = CountSemaphore(1)

    t1 = Thread(target=task1, args=(sem,))
    t2 = Thread(target=task2, args=(sem,))

    t1.start()
    time.sleep(1);
    t2.start()

    t1.join()
    t2.join()
```

# [**Unisex Bathroom Problem**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/unisex-bathroom-problem)

A synchronization practice problem requiring us to synchronize the usage of a single bathroom by both the genders.

## **Unisex Bathroom Problem**

A bathroom is being designed for the use of both males and females in an office but requires the following constraints to be maintained:

- There cannot be men and women in the bathroom at the same time.
- There should never be more than three employees in the bathroom simultaneously.

The solution should avoid deadlocks. For now, though, don’t worry about starvation.

```python
from threading import Thread
from threading import Semaphore
from threading import Condition
import time

class UnisexBathroomProblem():

    def __init__(self):
        self.in_use_by = "none"
        self.emps_in_bathroom = 0
        self.max_emps_sem = Semaphore(3)
        self.cond = Condition()

    def use_bathroom(self, name):
        # simulate using a bathroom
        print("\n{0} is using the bathroom. {1} employees in bathroom".format(name, self.emps_in_bathroom))
        time.sleep(1)
        print("\n{0} is done using the bathroom".format(name))

    def male_use_bathroom(self, name):

        with self.cond:
            while self.in_use_by == "female":
                self.cond.wait()
            self.max_emps_sem.acquire()
            self.emps_in_bathroom += 1
            self.in_use_by = "male"

        self.use_bathroom(name)
        self.max_emps_sem.release()

        with self.cond:
            self.emps_in_bathroom -= 1
            if self.emps_in_bathroom == 0:
                self.in_use_by = "none"

            self.cond.notifyAll()

    def female_use_bathroom(self, name):

        with self.cond:
            while self.in_use_by == "male":
                self.cond.wait()

            self.max_emps_sem.acquire()
            self.emps_in_bathroom += 1
            self.in_use_by = "female"

        self.use_bathroom(name)
        self.max_emps_sem.release()

        with self.cond:
            self.emps_in_bathroom -= 1

            if self.emps_in_bathroom == 0:
                self.in_use_by = "none"

            self.cond.notifyAll()

if __name__ == "__main__":
    problem = UnisexBathroomProblem()

    female1 = Thread(target=problem.female_use_bathroom, args=("Lisa",))
    male1 = Thread(target=problem.male_use_bathroom, args=("John",))
    male2 = Thread(target=problem.male_use_bathroom, args=("Bob",))
    female2 = Thread(target=problem.female_use_bathroom, args=("Natasha",))
    male3 = Thread(target=problem.male_use_bathroom, args=("Anil",))
    male4 = Thread(target=problem.male_use_bathroom, args=("Wentao",))
    male5 = Thread(target=problem.male_use_bathroom, args=("Nikhil",))
    male6 = Thread(target=problem.male_use_bathroom, args=("Paul",))
    male7 = Thread(target=problem.male_use_bathroom, args=("Klemond",))
    male8 = Thread(target=problem.male_use_bathroom, args=("Bill",))
    male9 = Thread(target=problem.male_use_bathroom, args=("Zak",))

    female1.start()
    male1.start()
    male2.start()
    time.sleep(1)
    female2.start()
    male3.start()
    male4.start()
    male5.start()
    male6.start()
    male7.start()
    male8.start()
    male9.start()
        

    female1.join()
    female2.join()
    male1.join()
    male2.join()
    male3.join()
    male4.join()
    male5.join()
    male6.join()
    male7.join()
    male8.join()
    male9.join()

    print("Employees in bathroom at the end {0}".format(problem.emps_in_bathroom))
```

# [**Implementing a Barrier**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/implementing-a-barrier)

This lesson discusses how a barrier can be implemented in Python.

## **Implementing a Barrier**

A barrier can be thought of as a point in the program code, which all or some of the threads need to reach at before any one of them is allowed to proceed further.

```python
from threading import Condition
from threading import Thread
from threading import current_thread
import time

class Barrier(object):
    def __init__(self, size):
        self.barrier_size = size
        self.reached_count = 0
        self.released_count = self.barrier_size
        self.cond = Condition()

    def arrived(self):

        self.cond.acquire()

        while self.reached_count == self.barrier_size:
            self.cond.wait()

        self.reached_count += 1

        if self.reached_count == self.barrier_size:
            self.released_count = self.barrier_size
        else:
            while self.reached_count < self.barrier_size:
                self.cond.wait()

        self.released_count -= 1

        if self.released_count == 0:
            self.reached_count = 0

        print("{0} released".format(current_thread().getName()), flush=True)
        self.cond.notifyAll()
        self.cond.release()

def thread_process(sleep_for):
    time.sleep(sleep_for)
    print("Thread {0} reached the barrier".format(current_thread().getName()), flush=True)
    barrier.arrived()

    time.sleep(sleep_for)
    print("Thread {0} reached the barrier".format(current_thread().getName()))
    barrier.arrived()

    time.sleep(sleep_for)
    print("Thread {0} reached the barrier".format(current_thread().getName()))
    barrier.arrived()

if __name__ == "__main__":
    barrier = Barrier(3)

    t1 = Thread(target=thread_process, args=(0,))
    t2 = Thread(target=thread_process, args=(0.5,))
    t3 = Thread(target=thread_process, args=(1.5,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
```

# [**Uber Ride Problem**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/uber-ride-problem)

This lesson solves the constraints of an imaginary Uber ride problem where Republicans and Democrats can't be seated as a minority in a four passenger car.

## **Uber Ride Problem**

Imagine at the end of a political conference, republicans and democrats are trying to leave the venue and ordering Uber rides at the same time. However, to make sure no fight breaks out in an Uber ride, the software developers at Uber come up with an algorithm whereby either an Uber ride can have all democrats or republicans or two Democrats and two Republicans. All other combinations can result in a fist-fight.

Your task as the Uber developer is to model the ride requestors as threads. Once an acceptable combination of riders is possible, threads are allowed to proceed to ride. Each thread invokes the method **`seated()`** when selected by the system for the next ride. When all the threads are seated, any one of the four threads can invoke the method **`drive()`** to inform the driver to start the ride.

```python
from threading import Thread
from threading import Semaphore
from threading import current_thread
from threading import Lock
from threading import Barrier
import random

class UberSeatingProblem():

    def __init__(self):
        self.democrats_count = 0
        self.democrats_waiting = Semaphore(0)
        self.republicans_count = 0
        self.republicans_waiting = Semaphore(0)
        self.lock = Lock()
        self.barrier = Barrier(4)
        self.ride_count = 0

    def drive(self):
        self.ride_count += 1
        print("Uber ride # {0} filled and on its way".format(self.ride_count), flush=True)

    def seated(self, party):
        print("\n{0} {1} seated".format(party, current_thread().getName()), flush=True)

    def seat_democrat(self):
        ride_leader = False

        self.lock.acquire()

        self.democrats_count += 1

        if self.democrats_count == 4:
            # release 3 democrats to ride along
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            ride_leader = True
            self.democrats_count -= 4

        elif self.democrats_count == 2 and self.republicans_count >= 2:
            # release 1 democrat and 2 republicans
            self.democrats_waiting.release()
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            ride_leader = True

            # remember to decrement the count of dems and repubs
            # selected for next ride
            self.democrats_count -= 2
            self.republicans_count -= 2

        else:
            # can't form a valid combination, keep waiting and release lock
            self.lock.release()
            self.democrats_waiting.acquire()

        self.seated("Democrat")
        self.barrier.wait()

        if ride_leader is True:
            self.drive()
            self.lock.release()

    def seat_republican(self):
        ride_leader = False

        self.lock.acquire()

        self.republicans_count += 1

        if self.republicans_count == 4:
            # release 3 republicans to ride along
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            ride_leader = True
            self.republicans_count -= 4

        elif self.republicans_count == 2 and self.democrats_count >= 2:
            # release 1 republican and 2 democrats
            self.republicans_waiting.release()
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            ride_leader = True

            # remember to decrement the count of dems and repubs
            # selected for next ride
            self.republicans_count -= 2
            self.democrats_count -= 2

        else:
            # can't form a valid combination, keep waiting and release lock
            self.lock.release()
            self.republicans_waiting.acquire()

        self.seated("Republican")
        self.barrier.wait()

        if ride_leader is True:
            self.drive()
            self.lock.release()

def random_simulation():
    problem = UberSeatingProblem()
    dems = 0
    repubs = 0

    riders = list()
    for _ in range(0, 16):
        toss = random.randint(0, 1)
        if toss == 1:
            riders.append(Thread(target=problem.seat_democrat))
            dems += 1
        else:
            riders.append(Thread(target=problem.seat_republican))
            repubs += 1

    print("Total {0} dems and {1} repubs".format(dems, repubs), flush=True)
    for rider in riders:
        rider.start()

    for rider in riders:
        rider.join()

def controlled_simulation():
    problem = UberSeatingProblem()
    dems = 10
    repubs = 10
    
    total = dems + repubs
    print("Total {0} dems and {1} repubs\n".format(dems, repubs))

    riders = list()

    while total is not 0:
        toss = random.randint(0, 1)
        if toss == 1 and dems is not 0:
            riders.append(Thread(target=problem.seat_democrat))
            dems -= 1
            total -= 1
        elif toss == 0 and repubs is not 0:
            riders.append(Thread(target=problem.seat_republican))
            repubs -= 1
            total -= 1

    for rider in riders:
        rider.start()

    for rider in riders:
        rider.join()

if __name__ == "__main__":
    controlled_simulation()

    # running the below simulation may hang the
    # program if an allowed combination can't be 
    # made
    #random_simulation()
```

# [**Dining Philosophers**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/dining-philosophers)

This chapter discusses the famous Dijkstra's Dining Philosopher's problem. Two different solutions are explained at length.

## **Dining Philosophers**

This is a classical synchronization problem proposed by Dijkstra.

Imagine you have five philosophers sitting on a roundtable. The philosopher's do only two kinds of activities. One: they contemplate, and two: they eat. However, they have only five forks between themselves to eat their food with. Each philosopher requires both the fork to his left and the fork to his right to eat his food.

The arrangement of the philosophers and the forks are shown in the diagram.

Design a solution where each philosopher gets a chance to eat his food without causing a deadlock.

```python
from threading import Thread
from threading import Semaphore
import random
import time

class DiningPhilosopherProblem:

    def __init__(self):
        self.forks = [None] * 5
        self.forks[0] = Semaphore(1)
        self.forks[1] = Semaphore(1)
        self.forks[2] = Semaphore(1)
        self.forks[3] = Semaphore(1)
        self.forks[4] = Semaphore(1)
        self.exit = False

    def life_cycle_of_a_philosopher(self, id):
        while self.exit is False:
            self.contemplate()
            self.eat(id)

    def contemplate(self):
        sleep_for = random.randint(800, 1200) / 1000
        time.sleep(sleep_for)

    def acquire_forks_for_right_handed_philosopher(self, id):
        self.forks[id].acquire()
        self.forks[(id + 1) % 5].acquire()

    def acquire_forks_for_left_handed_philosopher(self, id):
        self.forks[(id + 1) % 5].acquire()
        self.forks[id].acquire()

    def eat(self, id):

        # We randomly selected the philosopher with
        # id 3 as left-handed. All others must be
        # right-handed to avoid a deadlock.
        if id is 3:
            self.acquire_forks_for_left_handed_philosopher(3)
        else:
            self.acquire_forks_for_right_handed_philosopher(id)

        # eat to your heart's content
        print("Philosopher {0} is eating".format(id))

        # release forks for others to use
        self.forks[id].release()
        self.forks[(id + 1) % 5].release()

if __name__ == "__main__":

    problem = DiningPhilosopherProblem()

    philosophers = list()

    for id in range(0, 5):
        philosophers.append(Thread(target=problem.life_cycle_of_a_philosopher, args=(id,)))

    for philosopher in philosophers:
        philosopher.start()

    time.sleep(6)
    problem.exit = True

    for philosopher in philosophers:
        philosopher.join()
```

# [**Barber Shop**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/barber-shop)

This lesson visits the synchronization issues when programmatically modeling a hypothetical barber shop and how they can be solved using Python's concurrency primitives.

## **Barber Shop**

A similar problem appears in Silberschatz and Galvin's OS book, and variations of this problem exist in the wild.

A barbershop consists of a waiting room with ***n*** chairs, and a barber chair for giving haircuts. If there are no customers to be served, the barber goes to sleep. If a customer enters the barbershop and all chairs are occupied, then the customer leaves the shop. If the barber is busy, but chairs are available, then the customer sits in one of the free chairs. If the barber is asleep, the customer wakes up the barber. Write a program to coordinate the interaction between the barber and the customers.

```python
from threading import Thread
from threading import Semaphore
from threading import Lock
import time

class BarberShop:

    def __init__(self):
        self.total_chairs = 3
        self.waiting_customers = 0
        self.haircuts_given = 0
        self.lock = Lock()
        self.wait_for_customer_to_enter = Semaphore(0)
        self.wait_for_barber_to_get_ready = Semaphore(0)
        self.wait_for_barber_to_get_ready = Semaphore(0)
        self.wait_for_barber_to_cut_hair = Semaphore(0)
        self.wait_for_customer_to_leave = Semaphore(0)

    def customer_walks_in(self):
        with self.lock:
            if self.waiting_customers == self.total_chairs:
                print("Customer walks out, all chairs occupied")
                return

            self.waiting_customers += 1

        self.wait_for_customer_to_enter.release()
        self.wait_for_barber_to_get_ready.acquire()
        
        with self.lock:
            self.waiting_customers -= 1

        self.wait_for_barber_to_cut_hair.acquire()
        self.wait_for_customer_to_leave.release()

    def barber(self):

        while True:
            self.wait_for_customer_to_enter.acquire()
            self.wait_for_barber_to_get_ready.release()

            self.haircuts_given += 1

            print("Barber cutting hair ... {0}".format(self.haircuts_given))
            time.sleep(0.05)

            self.wait_for_barber_to_cut_hair.release()
            self.wait_for_customer_to_leave.acquire()

if __name__ == "__main__":

    barber_shop = BarberShop()

    barber_thread = Thread(target=barber_shop.barber)
    barber_thread.setDaemon(True)
    barber_thread.start()

    # intially 10 customers enter the barber shop one after the other
    customers = list()
    for _ in range(0, 10):
        customers.append(Thread(target=barber_shop.customer_walks_in))

    for customer in customers:
        customer.start()

    time.sleep(0.5)

    # second wave of 5 customers
    late_customers = list()
    for _ in range(0, 5):
        late_customers.append(Thread(target=barber_shop.customer_walks_in))

    for customer in late_customers:
        customer.start()

    for customer in customers:
        customer.join()

    for customer in late_customers:
        customer.join()
```

## [**Asynchronous to Synchronous Problem**](https://www.educative.io/courses/python-concurrency-for-senior-engineering-interviews/asynchronous-to-synchronous-problem)

*This is an actual interview question asked at Netflix.*

Imagine we have an **`AsyncExecutor`** class that performs some useful task asynchronously via the method **`execute()`**. In addition, the method accepts a function object that acts as a callback and gets invoked after the asynchronous execution is done. The definition for the involved classes is below. The asynchronous work is simulated using sleep. A passed-in call is invoked to let the invoker take any desired action after the asynchronous processing is complete.

```python
from threading import Thread
from threading import Condition
from threading import current_thread
import time

class AsyncExecutor:

    def work(self, callback):
        time.sleep(5)
        callback()

    def execute(self, callback):
        Thread(target=self.work, args=(callback,)).start()

class SyncExecutor(AsyncExecutor):

    def __init__(self):
        self.cv = Condition()
        self.is_done = False

    def work(self, callback):
        super().work(callback)

        print("{0} thread notifying".format(current_thread().getName()))
        self.cv.acquire()
        self.cv.notifyAll()
        self.is_done = True
        self.cv.release()

    def execute(self, callback):
        super().execute(callback)

        self.cv.acquire()
        while self.is_done is False:
            self.cv.wait()
        print("{0} thread woken-up".format(current_thread().getName()))
        self.cv.release()

def say_hi():
    print("Hi")

if __name__ == "__main__":
    exec = SyncExecutor()
    exec.execute(say_hi)

    print("main thread exiting")
```

## **Merge Sort**

Merge sort is a typical text-book example of a recursive algorithm and the poster-child of the divide and conquer strategy. The idea is very simple: we divide the array into two equal parts, sort them recursively, then combine the two sorted arrays. The base case for recursion occurs when the size of the array reaches a single element. An array consisting of a single element is already sorted.

The running time for a recursive solution is expressed as a *recurrence equation*. An equation or inequality that describes a function in terms of its own value on smaller inputs is called a recurrence equation. The running time for a recursive algorithm is the solution to the recurrence equation. The recurrence equation for recursive algorithms usually takes on the following form:

**Running Time = Cost to divide into n subproblems + n * Cost to solve each of the n problems + Cost to merge all n problems**

In the case of merge sort, we divide the given array into two arrays of equal size, i.e. we divide the original problem into sub-problems to be solved recursively.

Following is the recurrence equation for merge sort:

**Running Time = Cost to divide into 2 unsorted arrays + 2 * Cost to sort half the original array + Cost to merge 2 sorted arrays**

> T(n)=Cost to divide into 2 unsorted arrays+2∗T(n2)+Cost to merge 2 sorted arrayswhenn>1T(n)=Costtodivideinto2unsortedarrays+2∗T(2n​)+Costtomerge2sortedarrayswhenn>1
> 
> 
> T(n)=O(1)  when n=1*T*(*n*)=*O*(1)*whenn*=1
> 

Remember, the *solution* to the recurrence equation will be the *running time* of the algorithm on an input of size n. Without getting into the details of how we'll solve the recurrence equation, the running time of merge sort is

> O(nlgn)O(nlgn)
> 

where *n* is the size of the input array.

```python
import random
import math
from threading import Thread
scratch = None

def merge_sort(start, end, input):
    global scratch

    if start == end:
        return

    mid = start + math.floor((end - start) / 2)

    # sort first half
    worker1 = Thread(target=merge_sort, args=(start, mid, input))

    # sort second half
    worker2 = Thread(target=merge_sort, args=(mid + 1, end, input))

    worker1.start()
    worker2.start()
    worker1.join()
    worker2.join()

    # merge the two sorted arrays
    i = start
    j = mid + 1

    for k in range(start, end + 1):
        scratch[k] = input[k]

    k = start
    while k <= end:

        if i <= mid and j <= end:
            input[k] = min(scratch[i], scratch[j])

            if input[k] == scratch[i]:
                i += 1
            else:
                j += 1

        elif i <= mid and j > end:
            input[k] = scratch[i]
            i += 1
        else:
            input[k] = scratch[j]
            j += 1

        k += 1

def create_data(size):
    unsorted_list = [None] * size
    random.seed()

    for i in range(0, size):
        unsorted_list[i] = random.randint(0, 1000)

    return unsorted_list

def print_list(lst):
    end = len(lst)
    for i in range(0, end):
        print(lst[i], end=" ")

if __name__ == "__main__":
    size = 12
    scratch = [None] * size
    unsorted_list = create_data(size)

    print_list(unsorted_list)
    merge_sort(0, size - 1, unsorted_list)
    print("\n\n")
    print_list(unsorted_list)
```

# [**1117. Building H2O**](https://leetcode.com/problems/building-h2o/)

There are two kinds of threads: `oxygen` and `hydrogen`. Your goal is to group these threads to form water molecules.

There is a barrier where each thread has to wait until a complete molecule can be formed. Hydrogen and oxygen threads will be given `releaseHydrogen` and `releaseOxygen` methods respectively, which will allow them to pass the barrier. These threads should pass the barrier in groups of three, and they must immediately bond with each other to form a water molecule. You must guarantee that all the threads from one molecule bond before any other threads from the next molecule do.

In other words:

- If an oxygen thread arrives at the barrier when no hydrogen threads are present, it must wait for two hydrogen threads.
- If a hydrogen thread arrives at the barrier when no other threads are present, it must wait for an oxygen thread and another hydrogen thread.

We do not have to worry about matching the threads up explicitly; the threads do not necessarily know which other threads they are paired up with. The key is that threads pass the barriers in complete sets; thus, if we examine the sequence of threads that bind and divide them into groups of three, each group should contain one oxygen and two hydrogen threads.

Write synchronization code for oxygen and hydrogen molecules that enforces these constraints.

**Example 1:**

```
Input: water = "HOH"
Output: "HHO"
Explanation: "HOH" and "OHH" are also valid answers.

```

**Example 2:**

```
Input: water = "OOHHHH"
Output: "HHOHHO"
Explanation: "HOHHHO", "OHHHHO", "HHOHOH", "HOHHOH", "OHHHOH", "HHOOHH", "HOHOHH" and "OHHOHH" are also valid answers.
```

```python
"""
https://leetcode.com/problems/building-h2o/description/
There are two kinds of threads: oxygen and hydrogen. Your goal is to group these threads to form water molecules.

There is a barrier where each thread has to wait until a complete molecule can be formed. Hydrogen and oxygen threads will be given releaseHydrogen and releaseOxygen methods respectively, which will allow them to pass the barrier. These threads should pass the barrier in groups of three, and they must immediately bond with each other to form a water molecule. You must guarantee that all the threads from one molecule bond before any other threads from the next molecule do.
"""

from threading import Barrier, Semaphore, Thread

class H2O:

    def __init__(self):
        self.oxyzen_sem = Semaphore(1)
        self.hydrozen_sem = Semaphore(2)
        self.barrier = Barrier(3)

    def oxygen(self, releaseOxygen):

        self.oxyzen_sem.acquire()

        self.barrier.wait()

        releaseOxygen()

        self.oxyzen_sem.release()

    def hydrogen(self, releaseHydrogen):
        self.hydrozen_sem.acquire()

        self.barrier.wait()

        releaseHydrogen()

        self.hydrozen_sem.release()

# def releaseHydrogen():
#     print('H')

# def releaseOxygen():
#     print('O')
#
# h2o = H2O()
# # Simulating the formation of water molecules with threading
# threads = []
# water = "HOHHOHOHO"

# for atom in water:
#     if atom == 'H':
#         t = Thread(target=h2o.hydrogen, args=(releaseHydrogen,))
#     else:
#         t = Thread(target=h2o.oxygen, args=(releaseOxygen,))
#     threads.append(t)
#     t.start()

# for t in threads:
#     t.join()

```

# [**1115. Print FooBar Alternately](https://leetcode.com/problems/print-foobar-alternately/)`**

Suppose you are given the following code:

```
class FooBar {
  public void foo() {
    for (int i = 0; i < n; i++) {
      print("foo");
    }
  }

  public void bar() {
    for (int i = 0; i < n; i++) {
      print("bar");
    }
  }
}

```

The same instance of `FooBar` will be passed to two different threads:

- thread `A` will call `foo()`, while
- thread `B` will call `bar()`.

Modify the given program to output `"foobar"` `n` times.

**Example 1:**

```
Input: n = 1
Output: "foobar"
Explanation: There are two threads being fired asynchronously. One of them calls foo(), while the other calls bar().
"foobar" is being output 1 time.

```

**Example 2:**

```
Input: n = 2
Output: "foobarfoobar"
Explanation: "foobar" is being output 2 times.
```

```python
from threading import Condition

class FooBar:
    def __init__(self, n):
        self.n = n
        self.cond = Condition()
        self.turn = 'foo'

    def foo(self, printFoo: 'Callable[[], None]') -> None:
        
        for i in range(self.n):
            with self.cond:
                if self.turn == 'bar':
                    self.cond.wait()
            
                # printFoo() outputs "foo". Do not change or remove this line.
                printFoo()
                self.turn = 'bar'
                self.cond.notify_all()

    def bar(self, printBar: 'Callable[[], None]') -> None:
        
        for i in range(self.n):
            with self.cond:
                if self.turn == 'foo':
                    self.cond.wait()
                # printBar() outputs "bar". Do not change or remove this line.
                printBar()
                self.turn = 'foo'

                self.cond.notify_all()
```

# [**1188. Design Bounded Blocking Queue**](https://leetcode.com/problems/design-bounded-blocking-queue/)

Implement a thread-safe bounded blocking queue that has the following methods:

- `BoundedBlockingQueue(int capacity)` The constructor initializes the queue with a maximum `capacity`.
- `void enqueue(int element)` Adds an `element` to the front of the queue. If the queue is full, the calling thread is blocked until the queue is no longer full.
- `int dequeue()` Returns the element at the rear of the queue and removes it. If the queue is empty, the calling thread is blocked until the queue is no longer empty.
- `int size()` Returns the number of elements currently in the queue.

Your implementation will be tested using multiple threads at the same time. Each thread will either be a producer thread that only makes calls to the `enqueue` method or a consumer thread that only makes calls to the `dequeue` method. The `size` method will be called after every test case.

Please do not use built-in implementations of bounded blocking queue as this will not be accepted in an interview.

```python
from threading import Condition

class BoundedBlockingQueue(object):

    def __init__(self, capacity: int):
        self.max_capacity = capacity
        self.current_size = 0
        self.cond = Condition()
        self.q = []
        
    def enqueue(self, element: int) -> None:
        with self.cond:
            while self.current_size == self.max_capacity:
                self.cond.wait()

            self.q.append(element)
            self.current_size += 1
            self.cond.notify_all()

    def dequeue(self) -> int:
        with self.cond:
            while self.current_size == 0:
                self.cond.wait()
            
            item = self.q.pop(0)
            self.current_size -= 1
            self.cond.notify_all()
            return item
        
    def size(self) -> int:
        with self.cond:
            return self.current_size

```

### Circuit Breaker

```python
import time
import random
import threading

class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = 0
        self.condition = threading.Condition()

    def _is_timeout(self):
        return time.time() - self.last_failure_time > self.recovery_timeout

    def call(self, func, *args, **kwargs):
        with self.condition:
            # Wait if the circuit breaker is open and recovery timeout has not passed
            while self.state == 'OPEN':
                if self._is_timeout():
                    self.state = 'HALF_OPEN'
                    self.condition.notify_all()  # Notify threads waiting on the condition
                else:
                    self.condition.wait()  # Wait for the condition to change

            try:
                result = func(*args, **kwargs)
                self._reset()
                return result
            except Exception as e:
                self._record_failure()
                raise e

    def _reset(self):
        with self.condition:
            self.failure_count = 0
            self.state = 'CLOSED'
            self.condition.notify_all()  # Notify threads waiting on the condition

    def _record_failure(self):
        with self.condition:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                self.last_failure_time = time.time()
                self.condition.notify_all()  # Notify threads waiting on the condition

    def __str__(self):
        with self.condition:
            return f'CircuitBreaker(state={self.state}, failure_count={self.failure_count})'

# Example usage

def unreliable_service():
    # Simulating a service that has a 50% chance of failing
    if random.choice([True, False]):
        print("Service call succeeded")
        return "Success"
    else:
        raise Exception("Service call failed")

def worker(circuit_breaker):
    for _ in range(5):
        try:
            result = circuit_breaker.call(unreliable_service)
            print(result)
        except Exception as e:
            print(e)
        time.sleep(1)
        print(circuit_breaker)

def main():
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10)

    threads = [threading.Thread(target=worker, args=(circuit_breaker,)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

```