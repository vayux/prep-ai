# HLD

**Here's a list of all the headings 2 with their links in the document:**

**Consistent hashing⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠**

**Design of a Unique ID Generator⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠**

**Design of a Monitoring System⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Distributed Cache⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Pub-sub System⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Rate Limiter⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Blob Store⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**The Distributed Search⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Distributed Logging Service⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠**

**Distributed Task Scheduler⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠**

**Sharded Counters⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Design of Quora⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Design of Uber⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠​**

**Typeahead Suggestion System⁠[1](https://www.notion.so/HLD-1042cb43a46580f5ac17cd00aa23b15d?pvs=21)⁠**

[https://github.com/ashishps1/awesome-system-design-resources?tab=readme-ov-file](https://github.com/ashishps1/awesome-system-design-resources?tab=readme-ov-file)

[https://www.educative.io/blog/complete-guide-to-system-design](https://www.educative.io/blog/complete-guide-to-system-design)

## **Consistent hashing**

**Consistent hashing** is an effective way to manage the load over the set of nodes. In consistent hashing, we consider that we have a conceptual ring of hashes from 00 to n−1*n*−1, where n*n* is the number of available hash values. We use each node’s ID, calculate its hash, and map it to the ring. We apply the same process to requests. Each request is completed by the next node that it finds by moving in the clockwise direction in the ring.

Whenever a new node is added to the ring, the immediate next node is affected. It has to share its data with the newly added node while other nodes are unaffected. It’s easy to scale since we’re able to keep changes to our nodes minimal. This is because only a small portion of overall keys need to move. The hashes are randomly distributed, so we expect the load of requests to be random and distributed evenly on average on the ring.

Consider we have a conceptual ring of hashes from 0 to n-1, where n is the total number of hash values in the ring

**The primary benefit of consistent hashing is that as nodes join or leave, it ensures that a minimal number of keys need to move.** However, the request load isn’t equally divided in practice. Any server that handles a large chunk of data can become a bottleneck in a distributed system. That node will receive a disproportionately large share of data storage and retrieval requests, reducing the overall system performance. As a result, these are referred to as hotspots.

## [**Design of a Unique ID Generator**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-unique-id-generator)

The requirements for our system are as follows:

- **Uniqueness**: We need to assign unique identifiers to different events for identification purposes.
- **Scalability**: The ID generation system should generate at least a billion unique IDs per day.
- **Availability**: Since multiple events happen even at the level of nanoseconds, our system should generate IDs for all the events that occur.
- **64-bit numeric ID**: We restrict the length to 64 bits because this bit size is enough for many years in the future. Let’s calculate the number of years after which our ID range will wrap around.
    
    Total numbers available = 264264 = 1.8446744 x 10191019
    
    Estimated number of events per day = 1 billion = 109109
    
    Number of events per year = 365 billion = 365×109365×109
    
    Number of years to deplete identifier range = 264365×109365×109264​ = 50,539,024.8595 years
    
    64 bits should be enough for a unique ID length considering these calculations.
    

### **First solution: UUID[#](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-unique-id-generator#First-solution-UUID)**

A straw man solution for our design uses **UUIDs (universally unique IDs)**. This is a 128-bit number and it looks like 123*e*4567*e*89*b*12*d*3*a*456426614174000 in hexadecimal. It gives us about 10381038 numbers. UUIDs have different versions. We opt for version 4, which generates a pseudorandom number.

### **Second solution: using a database**

Let’s try mimicking the auto-increment feature of a database. Consider a central database that provides a current ID and then increments the value by one. We can use the current ID as a unique identifier for our events.

### **Third solution: using a range handler[#](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-unique-id-generator#Third-solution-using-a-range-handler)**

Let’s try to overcome the problems identified in the previous methods. We can use ranges in a central server. Suppose we have multiple ranges for one to two billion, such as 1 to 1,000,000; 1,000,001 to 2,000,000; and so on. In such a case, a central microservice can provide a range to a server upon request.

### **Lamport clocks**

In **Lamport clocks**, each node has its counter. All of the system’s nodes are equipped with a numeric counter that begins at zero when first activated. Before executing an event, the numeric counter is incremented by one. The message sent from this event to another node has the counter value. When the other node receives the message, it first updates its logical clock by taking the maximum of its clock value. Then, it takes the one sent in a message and then executes the message.

Lamport clocks provide a unique partial ordering of events using the happened-before relationship. We can also get a total ordering of events by tagging unique node/process identifiers, though such ordering isn’t unique and will change with a different assignment of node identifiers. However, we should note that Lamport clocks don’t allow us to infer causality at the global level. This means we can’t simply compare two clock values on any server to infer happened-before relationship. Vector clocks overcome this shortcoming.

### **Vector clocks**

Vector clocks maintain causal history—that is, all information about the happened-before relationships of events. So, we must choose an efficient data structure to capture the causal history of each event.

Consider the design shown below. We’ll generate our ID by concatenating relevant information, just like the Twitter snowflake, with the following division:

- **Sign bit**: A single bit is assigned as a sign bit, and its value will always be zero.
- **Vector clock**: This is 53 bits and the counters of each node.
- **Worker number**: This is 10 bits. It gives us 2^{10} = 1,024 worker IDs.

The following slides explain the unique ID generation using vector clocks, where the nodes A, B, and C reside in a data center.

`[vector-clock][worker-id]`

## [**Design of a Monitoring System**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-monitoring-system)

![Screenshot 2024-09-17 at 7.47.57 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_7.47.57_PM.png)

![Screenshot 2024-09-17 at 7.49.31 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_7.49.31_PM.png)

## **Distributed Cache**

- **Write-through cache**: The write-through mechanism writes on the cache as well as on the database. Writing on both storages can happen concurrently or one after the other.
- **Write-back cache**: In the write-back cache mechanism, the data is first written to the cache and asynchronously written to the database.
- **Write-around cache**: This strategy involves writing data to the database only. Later, when a read is triggered for the data, it’s written to cache after a cache miss.

![Screenshot 2024-09-17 at 7.55.54 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_7.55.54_PM.png)

![Screenshot 2024-09-17 at 7.57.20 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_7.57.20_PM.png)

## [**Pub-sub System**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-pub-sub-system)

![Screenshot 2024-09-17 at 8.01.16 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.01.16_PM.png)

## [**Rate Limiter**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-rate-limiter)

```python
domain: messaging
descriptors:
   -key: message_type
    value: marketing
    rate_limit:
               unit: day
               request_per_unit: 5
```

![Screenshot 2024-09-17 at 8.04.10 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.04.10_PM.png)

- Token bucket
- Leaking bucket
- Fixed window counter
- Sliding window log
- Sliding window counter

![Screenshot 2024-09-17 at 8.07.15 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.07.15_PM.png)

## [**Blob Store**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-considerations-of-a-blob-store)

## **Summary of the Lesson**

| **Section** | **Purpose** |
| --- | --- |
| Blob metadata | This is the metadata that’s maintained to ensure efficient storage and retrieval of blobs. |
| Partitioning | This determines how blobs are partitioned among different data nodes. |
| Blob indexing | This shows us how to efficiently search for blobs. |
| Pagination | This teaches us how to conceive a method for the retrieval of a limited number of blobs to ensure improved readability and loading time. |
| Replication | This teaches us how to replicate blobs and tells us how many copies we should maintain to improve availability. |
| Garbage collection | This teaches us how to delete blobs without sacrificing performance. |
| Streaming | This teaches us how to stream large files chunk-by-chunk to facilitate interactivity for users. |
| Caching | This shows us how to improve response time and throughput. |

![Screenshot 2024-09-17 at 8.11.45 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.11.45_PM.png)

## [**The Distributed Search**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/system-design-the-distributed-search)

A **search system** is a system that takes some text input, a search query, from the user and returns the relevant content in a few seconds or less. There are three main components of a search system, namely:

- A **crawler**, which fetches content and creates documents.
- An **indexer**, which builds a searchable index.
- A **searcher**, which responds to search queries by running the search query on the *index* created by the *indexer*.

### **Inverted index**

An **inverted index** is a HashMap-like data structure that employs a document-term matrix. Instead of storing the complete document as it is, it splits the documents into individual words. After this, the **document-term matrix** identifies unique words and discards frequently occurring words like “to,” “they,” “the,” “is,” and so on. Frequently occurring words like those are called **terms**. The document-term matrix maintains a **term-level index** through this identification of unique words and deletion of unnecessary terms.

**Inverted index** is one of the most popular index mechanisms used in document retrieval. It enables efficient implementation of boolean, extended boolean, proximity, relevance, and many other types of search algorithms.

![Screenshot 2024-09-17 at 8.19.16 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.19.16_PM.png)

![Screenshot 2024-09-17 at 8.21.50 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.21.50_PM.png)

![Screenshot 2024-09-17 at 8.22.11 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.22.11_PM.png)

## [**Distributed Logging Service**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/design-of-a-distributed-logging-service)

![Screenshot 2024-09-17 at 8.25.29 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.25.29_PM.png)

## **Distributed Task Scheduler**

![Screenshot 2024-09-17 at 8.40.35 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.40.35_PM.png)

![Screenshot 2024-09-17 at 8.42.11 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.42.11_PM.png)

## [**Sharded Counters**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/detailed-design-of-sharded-counters)

![Screenshot 2024-09-17 at 8.44.27 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.44.27_PM.png)

## [**Design of Quora**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/final-design-of-quora)

![Screenshot 2024-09-17 at 8.47.16 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.47.16_PM.png)

## [**Design of Uber**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/detailed-design-of-uber)

![Screenshot 2024-09-17 at 8.49.58 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.49.58_PM.png)

## [**Typeahead Suggestion System**](https://www.educative.io/courses/grokking-modern-system-design-interview-for-engineers-managers/detailed-design-of-the-typeahead-suggestion-system)

![Screenshot 2024-09-17 at 8.51.21 PM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-17_at_8.51.21_PM.png)

### [LSM Tree](https://www.educative.io/courses/deep-dive-into-the-internals-of-the-database/log-structured-merge-tree)

## **Introduction**

The **Log Structured Merge Tree (LSM)** is a disk-resident data structure to persist key-value pairs optimized for write-heavy query patterns.

LSM includes an append-only storage structure as the primary data structure to handle high write volumes. These storage structures are periodically merged and compacted in the background to eliminate duplicate and deleted records. LSM is a prominent data structure used in multiple NoSQL datastores such as Cassandra and Hbase.

## **Data structure**

LSM includes four main data structures:

- Memtable
- SSTable
- WAL
- Bloom filter

### **Memtable**

- **Memtable** is an in-memory data structure
- acts as the frontier for all the client requests handling reads and writes.
- a balanced binary search tree such as AVL tree,
- guaranteeing that the time complexity of insert, delete, update, and read requests be O(log2M)*O*(*log*2*M*).

### **SSTable**

- **Sorted String Tables (SSTable)**
- on-disk data structures that are a collection of files referred to as **segment files**.
- Each segment file stores multiple key-value pairs sorted by key.
- These segment files are periodically scanned in the background and merged to create large segment files.
- The resulting segment files eliminate, duplicate, and deleted records.

Each segment file includes an additional index structure to speed up lookups in the file system implemented in B+ tree.

Since SSTables store key-value pairs sorted by key, the implementation of SSTables groups multiple key-value pairs into a block and compresses them for efficient storage. This compression enables Memtable to persist only spatial keys leading to range scans.

[](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNTkxIiBoZWlnaHQ9IjI5MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2ZXJzaW9uPSIxLjEiLz4=)

Structure of the SSTable

In the SSTable above:

- Segment 1 includes key ranges from `1000` to `1099`.
- Segment 2 includes key ranges from `2000` to `2999` .
- Segment 3 includes key ranges from `3000` to `3999`.

### **WAL**

Memtable acts as a frontier for client requests. All the insert, update, and delete requests are batched and synced with SSTable periodically. Memtable is an in-memory data structure and is subject to the problem of volatility on process restarts. To ensure durability, Memtable is accompanied by a disk-resident, append-only data structure WAL.

Every modification request is applied on Memtable and appended to WAL before the client considers the modification request successful.

### **Bloom filter**

The ****Bloom filter is a probabilistic space-efficient data structure used to check the existence of a particular key-value pair without querying the underlying on-disk data structure. 

A Bloom filter **can accurately verify if a specific key** is not a set member and probabilistically verify if the key is part of the set.

- A **false positive means that the assumption of a result is true but not valid.** For example, if the Bloom filter determines the element is present, and the corresponding data is missing in the actual database, it is a false positive.
- A **false negative match means that the assumption of a result is false but not valid.** For example, if the Bloom filter determines the element is missing and the corresponding data is present in the actual database, it is a false negative.

Since the Memtable has limited capacity, the read requests to LSM go through Memtable, followed by multiple segment files. Bloom filter avoids multiple lookups in the case of missing keys. **A Bloom filter can return a false positive but can never produce a false negative**.

![Screenshot 2024-09-18 at 12.57.23 AM.png](HLD%201042cb43a46580f5ac17cd00aa23b15d/Screenshot_2024-09-18_at_12.57.23_AM.png)

## **Compaction**

The number of SSTables increases exponentially over a period of time, and affects the read performance. In addition, the keys are duplicated across multiple SSTables, making the read less performant and increasing disk size. Therefore, 

LSM includes a process of merging and compaction to reconcile SSTables and eliminate duplicates.

Every SSTable and, in turn, every segment file inside SSTable, has data sorted by key. A **multiway merge sort algorithm merges multiple SSTables**, eliminating duplicate keys and retaining only the latest key and value.

The compaction process creates a priority queue such as **Min Heap** that can hold up to N elements