FROM python:3.12-slim

# Install pdm 
RUN curl -sSL https://pdm-project.org/install-pdm.py | python3 -
RUN echo $(pdm --version)

# Run code
RUN mkdir -p /workdir
COPY . /workdir
WORKDIR /workdir
RUN pdm install

EXPOSE 8501
# RUN python -m token_trace.load_pretrained_model
CMD ["streamlit", "run", "src/token_trace_demo/app/run_app.py"]