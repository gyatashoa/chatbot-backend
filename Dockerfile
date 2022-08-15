FROM python
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
EXPOSE 8000