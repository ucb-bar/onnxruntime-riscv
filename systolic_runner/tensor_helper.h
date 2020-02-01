#include <queue>
typedef std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_pq;

min_pq getTopK(float* arr, int length, int k) {
    min_pq pq;
    for (int i = 0; i < length; i++) {
        if (pq.size() <= k) {
            pq.push(std::make_pair(arr[i], i));
        } else if (arr[i] > pq.top().first) {
            pq.pop();
            pq.push(std::make_pair(arr[i], i));
        }
    }
    return pq;
}