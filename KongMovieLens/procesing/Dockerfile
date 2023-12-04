# Define a base stage that uses the official python runtime base image
FROM python:3.11-slim AS base

# Add curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set the application directory
WORKDIR /usr/local/app

# Install our requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[dataframe]" --upgrade  

# Define a stage specifically for development, where it'll watch for
# filesystem changes
FROM base AS dev
RUN pip install watchdog
ENV FLASK_ENV=development
CMD ["python", "app.py"]

# Define the final stage that will bundle the application for production
FROM base AS final

# Copy our code from the current folder to the working directory inside the container
COPY . .

# Make port 80 available for links and/or publish
EXPOSE 80

# Define our command to be run when launching the container
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:80", "--log-file", "-", "--access-logfile", "-", "--workers", "4", "--keep-alive", "0"]







# Define a base stage that uses the official python runtime base image
#FROM python:3.11-slim AS base

# Add curl for healthcheck
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends curl wget unzip && \
#    rm -rf /var/lib/apt/lists/*

# Set the application directory
#WORKDIR /usr/local/app

# Install our requirements.txt
#COPY requirements.txt ./requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install "dask[dataframe]" --upgrade  
#RUN wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
#RUN unzip ml-10m.zip
#RUN rm ml-10m.zip
#RUN mv ml-10M100K/ratings.dat .
#RUN rm -rf ml-10M1100K
#RUN sed -i 's/::/\t/g' ratings.dat

# Define a stage specifically for development, where it'll watch for
# filesystem changes
#FROM base AS dev
#RUN pip install watchdog
#ENV FLASK_ENV=development
#CMD ["python", "app.py"]

# Define the final stage that will bundle the application for production
#FROM base AS final

# Copy our code from the current folder to the working directory inside the container
#COPY . .

# Make port 80 available for links and/or publish
#EXPOSE 80

# Define our command to be run when launching the container
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:80", "--log-file", "-", "--access-logfile", "-", "--workers", "4", "--keep-alive", "0"]

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends curl wget unzip  && \
#    rm -rf /var/lib/apt/lists/*



#RUN pip install "dask[dataframe]" --upgrade  
#RUN wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
#RUN unzip ml-10m.zip
#RUN rm ml-10m.zip
#RUN mv ml-10M100K/ratings.dat .
#RUN rm -rf ml-10M1100K
#RUN sed -i 's/::/\t/g' ratings.dat


