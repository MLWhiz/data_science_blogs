# How to Use FastAPI

To run this code;

1. Install Fastapi and uvicorn using:

```
pip install fastapi
pip install uvicorn
```

2. Clone this repo.

```
git clone https://github.com/MLWhiz/data_science_blogs
cd data_science_blogs/fastapi
```

3. To run the GET Example, run on terminal: 
```
$ uvicorn fastapiapp:app --reload
```
4. To run the PUT Example, run on terminal:
```
$ uvicorn fastapi_put:app --reload
```

5. Open `http://127.0.0.1:8000/docs` in browser to use GUI to test the API. 

For more details,check out this blog:
[A layman guide for Data Scientists to create API in minutes](https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f)