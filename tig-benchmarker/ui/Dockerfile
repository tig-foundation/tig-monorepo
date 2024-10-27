FROM node:latest AS build

WORKDIR /app

COPY . .

RUN npm install -g @angular/cli
RUN npm install --legacy-peer-deps

ARG CONFIG

RUN ng build

FROM nginx:latest

COPY --from=build /app/dist/tig-brenchmarker-ui/browser /usr/share/nginx/html
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
