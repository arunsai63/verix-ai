# VerixAI Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key
- At least 8GB RAM available
- 20GB free disk space

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/verix-ai.git
cd verix-ai
```

### 2. Set Up Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-actual-api-key-here
SECRET_KEY=generate-a-secure-random-key
```

### 3. Start with Docker Compose

#### Development Mode
```bash
docker-compose up -d
```

#### Production Mode
```bash
docker-compose --profile production up -d
```

### 4. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Manual Installation

### Backend Setup

1. **Create Python Virtual Environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the Backend**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install Node Dependencies**
```bash
cd frontend
npm install
```

2. **Configure API Endpoint**
Create `.env.local`:
```
REACT_APP_API_URL=http://localhost:8000
```

3. **Run the Frontend**
```bash
npm start
```

## Production Deployment

### Using Docker

1. **Build Production Images**
```bash
docker-compose -f docker-compose.prod.yml build
```

2. **Run Production Stack**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Cloud Deployment

#### AWS EC2

1. Launch an EC2 instance (t3.large or larger recommended)
2. Install Docker and Docker Compose
3. Clone the repository
4. Configure environment variables
5. Run docker-compose

#### Google Cloud Run

1. Build and push images to Google Container Registry
2. Deploy backend as Cloud Run service
3. Deploy frontend to Cloud Storage + CDN
4. Configure Cloud SQL for PostgreSQL

#### Azure Container Instances

1. Create Azure Container Registry
2. Push Docker images
3. Deploy using Azure Container Instances
4. Configure Azure Database for PostgreSQL

## SSL/TLS Configuration

### Using Let's Encrypt with Nginx

1. **Install Certbot**
```bash
sudo apt-get install certbot python3-certbot-nginx
```

2. **Obtain Certificate**
```bash
sudo certbot --nginx -d your-domain.com
```

3. **Update nginx.conf**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    # ... rest of configuration
}
```

## Database Management

### Backup
```bash
docker exec verixai-postgres pg_dump -U verixai verixai_db > backup.sql
```

### Restore
```bash
docker exec -i verixai-postgres psql -U verixai verixai_db < backup.sql
```

### Migration
```bash
docker exec verixai-backend alembic upgrade head
```

## Monitoring

### Health Checks
- Backend: `GET /health`
- Frontend: Check port 3000
- PostgreSQL: `pg_isready`
- ChromaDB: `GET :8001/api/v1/heartbeat`

### Logging
```bash
# View backend logs
docker logs verixai-backend -f

# View all logs
docker-compose logs -f
```

### Metrics
Consider integrating:
- Prometheus for metrics collection
- Grafana for visualization
- Sentry for error tracking

## Scaling

### Horizontal Scaling

1. **Backend Workers**
```yaml
backend:
  deploy:
    replicas: 3
```

2. **Load Balancing**
Add nginx or HAProxy for load balancing

3. **Database Replication**
Configure PostgreSQL streaming replication

### Vertical Scaling
Increase container resources:
```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000
# Kill the process or change the port in docker-compose.yml
```

2. **Database Connection Failed**
```bash
# Check PostgreSQL is running
docker ps | grep postgres
# Check logs
docker logs verixai-postgres
```

3. **OpenAI API Errors**
- Verify API key is correct
- Check API quota and limits
- Ensure network connectivity

4. **Out of Memory**
```bash
# Increase Docker memory limit
docker system prune -a  # Clean up first
# Then adjust in Docker Desktop settings
```

## Security Considerations

1. **Environment Variables**
- Never commit `.env` files
- Use secrets management in production
- Rotate keys regularly

2. **Network Security**
- Use firewall rules
- Implement rate limiting
- Enable CORS properly

3. **Data Protection**
- Encrypt data at rest
- Use SSL/TLS for all connections
- Implement access controls

## Maintenance

### Regular Tasks
- Update dependencies monthly
- Backup database daily
- Monitor disk usage
- Review logs for errors
- Update SSL certificates

### Updates
```bash
# Update backend
cd backend
pip install -r requirements.txt --upgrade

# Update frontend
cd frontend
npm update

# Update Docker images
docker-compose pull
docker-compose up -d
```

## Support

For issues and questions:
- GitHub Issues: [your-repo/issues]
- Documentation: [/docs]
- Email: support@verixai.com