# Stage 1: Build stage (jika ada assets yang perlu diproses)
FROM node:18-alpine AS builder

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies jika ada (optional)
# RUN npm install

# Stage 2: Production stage menggunakan Nginx
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy aplikasi dari stage builder
COPY --from=builder /app /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/index.html || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
