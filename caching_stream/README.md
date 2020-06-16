In the new release notes for 0.57.0 which just came out yesterday, streamlit has made updates to st.cache. One notable change to this release is the “ability to set expiration options for cached functions by setting the max_entries and ttl arguments”. From the documentation:

max_entries (int or None) — The maximum number of entries to keep in the cache, or None for an unbounded cache. (When a new entry is added to a full cache, the oldest cached entry will be removed.) The default is None.

ttl (float or None) — The maximum number of seconds to keep an entry in the cache, or None if cache entries should not expire. The default is None.

Two use cases where this might help would be:
If you’re serving your app and don’t want the cache to grow forever.
If you have a cached function that reads live data from a URL and should clear every few hours to fetch the latest data
So now what you need to do is just:

```
@st.cache(ttl=60*5,max_entries=20)
def hit_news_api(country, n):
    # hit_api
```

[More Here](https://towardsdatascience.com/advanced-streamlit-caching-6f528a0f9993)