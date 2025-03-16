# Technical Context: Web Research Assistant & Report Writer

## Technology Stack

### Core Technologies
- **Python 3.10+**: Primary implementation language
- **FastAPI/Flask**: API and web services framework
- **Docker**: Containerization for deployment
- **MongoDB**: Document storage for research data and reports
- **ChromaDB/Pinecone**: Vector database for semantic storage
- **Redis**: Caching and task management

### LLM Integration
- **GPT-4o**: Primary model for reasoning and content generation
- **OpenAI API**: Main interface for LLM integration
- **LangChain**: Framework for building LLM applications
- **DSPy**: Structured output generation
- **LlamaIndex**: Primary framework for RAG capabilities and workflow orchestration

### Web Interaction
- **Playwright/Selenium**: Web browsing and content extraction
- **BeautifulSoup**: HTML parsing and data extraction
- **SerpAPI/Tavily**: Search engine interaction

### Document Processing
- **Pandoc**: Document format conversion
- **PyPDF2**: PDF handling and generation
- **python-docx**: Word document generation
- **Markdown**: Internal document representation

### Monitoring and Observability
- **Logging**: Structured logging for system events
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Safety and Validation
- **NeMo Guardrails**: Content safety checks
- **Pydantic**: Data validation and schema enforcement
- **RegEx**: Pattern matching for sensitive information detection

## Development Environment

### Local Development
- **IDE**: VS Code with Python extensions
- **Version Control**: Git with GitHub
- **Virtual Environment**: Poetry for dependency management
- **Testing**: pytest for unit and integration testing
- **Code Quality**: Black, isort, flake8, mypy
- **Pre-commit Hooks**: Automated style and lint checks

### CI/CD Pipeline
- **GitHub Actions**: Automated testing and deployment
- **Docker Compose**: Multi-container application orchestration
- **CodeCov**: Test coverage tracking
- **Dependabot**: Dependency updates and security alerts

## Deployment Architecture

### Container Orchestration
```yaml
# Key services in the deployment
services:
  api-gateway:        # Entry point for all client requests
  user-service:       # User management and authentication
  research-service:   # Core research orchestration
  report-service:     # Report generation and formatting
  llm-service:        # LLM interaction and caching
  vector-db:          # Vector database for semantic search
  storage-service:    # Document storage and retrieval
  mongodb:            # Persistent document storage
  redis:              # Cache and message broker
```

### Deployment Options
1. **Self-hosted**: 
   - Docker Compose for small-scale deployments
   - Kubernetes for production-scale deployments
   
2. **Cloud Deployment**:
   - AWS ECS/EKS for container orchestration
   - AWS S3 for document storage
   - MongoDB Atlas for managed database
   - AWS Lambda for serverless components

## APIs and Integration Points

### External API Dependencies
- **OpenAI API**: Core LLM capabilities
  - Authentication: API Key
  - Rate Limits: Varies by tier
  - Endpoint: https://api.openai.com/v1/
  
- **SerpAPI**: Web search capabilities
  - Authentication: API Key
  - Rate Limits: Per subscription
  - Endpoint: https://serpapi.com/search
  
- **Claude API (Optional)**: Secondary LLM
  - Authentication: API Key
  - Rate Limits: Per subscription
  - Endpoint: https://api.anthropic.com/v1/

### Internal APIs
- **Research API**: `/api/v1/research`
  - POST `/create`: Start new research
  - GET `/status/{id}`: Check research status
  - GET `/results/{id}`: Get research results
  
- **Report API**: `/api/v1/reports`
  - GET `/{id}`: Get report by ID
  - GET `/{id}/download`: Download formatted report
  - POST `/{id}/feedback`: Submit feedback

## Technical Constraints

### Performance Requirements
- Maximum research time: 15 minutes for standard topics
- Maximum memory usage: 2GB per research session
- Maximum storage per report: 50MB
- API response time: <500ms for status checks

### Scalability Considerations
- Stateless design for horizontal scaling
- Asynchronous processing for long-running tasks
- Caching for repeated queries and LLM calls
- Connection pooling for database efficiency

### Security Requirements
- API authentication using JWT
- HTTPS for all communications
- Input validation for all endpoints
- Rate limiting to prevent abuse
- PII detection and sanitization

## Dependencies and Libraries

### Core Dependencies
```
langchain>=0.1.0
openai>=1.3.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
chromadb>=0.4.0
redis>=4.6.0
pymongo>=4.5.0
pytest>=7.4.0
```

### Web Interaction
```
playwright>=1.40.0
beautifulsoup4>=4.12.0
requests>=2.31.0
serpapi>=0.1.0
tavily-python>=0.1.0
```

### Document Processing
```
pypdf>=3.16.0
python-docx>=0.8.11
markdown>=3.5.0
pandoc>=2.3
```

### Monitoring and Validation
```
prometheus-client>=0.17.0
nemoguardrails>=0.4.0
pydantic>=2.0.0
```

## Technical Debt and Considerations

### Current Limitations
- Search results limited by third-party API constraints
- Document format support limited to common formats
- No streaming for long-running research tasks
- Limited offline capability

### Future Technical Enhancements
- Streaming research results for progressive updates
- Enhanced caching for improved performance
- Local file analysis capabilities
- Support for additional output formats
- Multi-modal content processing
- Fine-tuned domain-specific models

## Development Guidelines

### Coding Standards
- PEP 8 compliance for Python code
- Type annotations for all functions
- Comprehensive docstrings in Google format
- Unit tests with >80% coverage
- Integration tests for critical paths

### Architecture Principles
1. **Modularity**: Components should have high cohesion, low coupling
2. **Testability**: Design for testability from the ground up
3. **Observability**: Every component should emit appropriate logs and metrics
4. **Resilience**: Design for graceful degradation
5. **Security**: Apply security by design principles
