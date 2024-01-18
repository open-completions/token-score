#include <iostream>
#include <unordered_map>
#include <list>
#include <stdexcept>

template<typename KeyType, typename ValueType>
class LRUCache {
private:
    size_t capacity;
    std::list<KeyType> itemsList;
    std::unordered_map<KeyType, std::pair<ValueType, typename std::list<KeyType>::iterator>> itemsMap;

public:
    LRUCache(size_t capacity) : capacity(capacity) {
        if (capacity <= 0) {
            throw std::invalid_argument("Cache capacity must be greater than 0");
        }
    }

    void put(const KeyType& key, const ValueType& value) {
        auto item = itemsMap.find(key);
        if (item != itemsMap.end()) {
            itemsList.erase(item->second.second);
            itemsList.push_front(key);
            item->second = { value, itemsList.begin() };
        } else {
            if (itemsMap.size() == capacity) {
                itemsMap.erase(itemsList.back());
                itemsList.pop_back();
            }
            itemsList.push_front(key);
            itemsMap[key] = { value, itemsList.begin() };
        }
    }

    ValueType get(const KeyType& key) {
        auto item = itemsMap.find(key);
        if (item == itemsMap.end()) {
            throw std::out_of_range("Key not found");
        }
        itemsList.erase(item->second.second);
        itemsList.push_front(key);
        item->second.second = itemsList.begin();
        return item->second.first;
    }

    bool exists(const KeyType& key) const {
        return itemsMap.find(key) != itemsMap.end();
    }

    size_t size() const {
        return itemsMap.size();
    }
};

int main() {
    LRUCache<int, std::string> cache(5);
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");
    try {
        std::cout << "Value for key 3: " << cache.get(3) << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
