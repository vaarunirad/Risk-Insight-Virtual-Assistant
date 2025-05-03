FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt for dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./

# Create model directories
RUN mkdir -p .models/time_series .models/isolation_forest

# Set Python to not buffer memory and optimize garbage collection
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PYTHONGC="aggressive"

# Add better error logging and debugging support with memory management
RUN echo '#!/bin/bash\n\
echo "===== COGO RISK INSIGHT PIPELINE ====="\n\
echo "Container starting at $(date)"\n\
echo "Environment variables:"\n\
env | grep -v "KEY\|PASSWORD\|SECRET" | sort\n\
echo "Python version: $(python --version)"\n\
echo "Files in container:"\n\
ls -la\n\
echo "System memory:"\n\
free -m\n\
echo ""\n\
echo "** Starting pipeline **"\n\
# Run with lower memory buffer settings\n\
PYTHONMALLOC=malloc python -m memory_profiler -c "import pipeline_runner; pipeline_runner.runpipeline()"\n\
exit_code=$?\n\
echo ""\n\
echo "Pipeline exited with code: $exit_code"\n\
echo "Final memory usage:"\n\
free -m\n\
if [ $exit_code -ne 0 ]; then\n\
  echo "Pipeline execution failed. Check logs for details."\n\
  sleep 120\n\
fi\n\
' > /app/start.sh && chmod +x /app/start.sh

# Add a simplified script to run without memory profiler if needed
RUN echo '#!/bin/bash\n\
echo "===== COGO RISK INSIGHT PIPELINE (SIMPLE MODE) ====="\n\
echo "Container starting at $(date)"\n\
echo "System memory:"\n\
free -m\n\
echo "** Starting pipeline **"\n\
PYTHONMALLOC=malloc python -c "import pipeline_runner; pipeline_runner.runpipeline()"\n\
exit_code=$?\n\
echo "Pipeline exited with code: $exit_code"\n\
echo "Final memory usage:"\n\
free -m\n\
' > /app/run_simple.sh && chmod +x /app/run_simple.sh

# Install memory-profiler for memory optimization
RUN pip install --no-cache-dir memory-profiler psutil

# Use the simple script by default
CMD ["/app/run_simple.sh"]