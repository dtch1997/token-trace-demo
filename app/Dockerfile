FROM python:3.12-slim

# Install dependencies
RUN python -m pip install --upgrade pip

# Run code
RUN mkdir -p /workdir
WORKDIR /workdir
COPY . /workdir
RUN python -m pip --timeout 1000 install -e .

EXPOSE 8501
# RUN python -m token_trace.load_pretrained_model
CMD ["streamlit", "run", "app/token_trace_app.py"]