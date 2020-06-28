# Deployment could be easy: A Data Scientist's Guide to deploy an Image detection FastAPI API using Amazon ec2
### Using Amazon EC2+Pytorch+Fastapi and Docker

Just recently, I had written a simple [tutorial on  FastAPI](https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f) which was about simplifying and understanding how APIs work, and creating a simple API using the framework. 
That post got quite a good response but the most asked question was how to deploy the FastAPI API on ec2 and how to use images data rather than simple strings, integers, and floats as input to the API.
I actually scoured the net for this,  but all I could find was some undercooked documentation and a lot of different ways people were taking to deploy using NGINX or ECS. None of those seemed particularly great or complete to me.
So, I tried to do this myself using some help from FastAPI documentation. In this post we will look at predominantly 4 things:
- Setting Up an Amazon Instance
- Creating a FastAPI API for Object Detection
- Deploying FastAPI using Docker
- An End to End App with UI

[Read More](https://towardsdatascience.com/deployment-could-be-easy-a-data-scientists-guide-to-deploy-an-image-detection-fastapi-api-using-329cdd80400)