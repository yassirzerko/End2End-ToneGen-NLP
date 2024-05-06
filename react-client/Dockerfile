FROM node:22.1-bullseye-slim

WORKDIR /app

COPY ./react-client ./react-client 

WORKDIR /app/react-client

RUN npm install -g serve

RUN npm install

RUN npm run build

# Expose the port specified by the build argument
EXPOSE 3000

CMD ['serve','-s', 'build', '-l', 3000]

