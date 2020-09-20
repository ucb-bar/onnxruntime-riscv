#include <queue>
template <typename T>
using min_pq = std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int>>, std::greater<std::pair<T, int>>> ;

template <typename T>
min_pq<T> getTopK(T* arr, int length, size_t k) {
    min_pq<T> pq;
    for (int i = 0; i < length; i++) {
        if (pq.size() < k) {
            pq.push(std::make_pair(arr[i], i));
        } else if (arr[i] > pq.top().first) {
            pq.pop();
            pq.push(std::make_pair(arr[i], i));
        }
    }
    return pq;
}