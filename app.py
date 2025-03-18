from flask import Flask, request, render_template, jsonify
import os
import sys
import time
import boto3
import uuid
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
import json

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
session = boto3.session.Session()
region = session.region_name
s3_client = boto3.client('s3')
sts_client = boto3.client('sts')
bedrock_agent_client = boto3.client('bedrock-agent')
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
account_id = sts_client.get_caller_identity()["Account"]

# KB settings
timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))[-7:]
suffix = f"{timestamp_str}"
knowledge_base_name = f"bedrock-flask-kb-{suffix}"
data_bucket_name = f'bedrock-kb-data-{suffix}-{account_id}'
foundation_model = "anthropic.claude-3-sonnet-20240229-v1:0"

# Knowledge Base setup
class BedrockKnowledgeBase:
    def __init__(self, kb_name, kb_description, data_sources, chunking_strategy="FIXED_SIZE", suffix=""):
        self.kb_name = kb_name
        self.kb_description = kb_description
        self.data_sources = data_sources
        self.chunking_strategy = chunking_strategy
        self.suffix = suffix
        self.kb_id = None
        self._setup_knowledge_base()
        
    def _setup_knowledge_base(self):
        # Create S3 buckets
        for source in self.data_sources:
            if source["type"] == "S3":
                try:
                    s3_client.create_bucket(Bucket=source["bucket_name"])
                    logger.info(f"Created S3 bucket: {source['bucket_name']}")
                except Exception as e:
                    logger.error(f"Error creating bucket: {e}")
                    
        # Create Knowledge Base
        try:
            response = bedrock_agent_client.create_knowledge_base(
                name=self.kb_name,
                description=self.kb_description,
                roleArn=self._create_role_for_kb(),
                knowledgeBaseConfiguration={
                    'type': 'VECTOR',
                    'vectorKnowledgeBaseConfiguration': {
                        'embeddingModelArn': f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v1"
                    }
                }
            )
            self.kb_id = response['knowledgeBase']['knowledgeBaseId']
            logger.info(f"Created Knowledge Base with ID: {self.kb_id}")
            
            # Add data sources
            for source in self.data_sources:
                if source["type"] == "S3":
                    self._add_s3_data_source(source["bucket_name"])
            
            return self.kb_id
            
        except Exception as e:
            logger.error(f"Error creating Knowledge Base: {e}")
            raise
    
    def _create_role_for_kb(self):
        # Create IAM role for Knowledge Base (simplified)
        # In a production environment, use more restrictive policies
        role_name = f"BedrockKBRole-{self.suffix}"
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        iam_client = boto3.client('iam')
        try:
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            
            # Attach necessary policies
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            f"arn:aws:s3:::{data_bucket_name}",
                            f"arn:aws:s3:::{data_bucket_name}/*"
                        ]
                    }
                ]
            }
            
            policy_name = f"BedrockKBPolicy-{self.suffix}"
            policy_response = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_document)
            )
            
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_response['Policy']['Arn']
            )
            
            return response['Role']['Arn']
            
        except Exception as e:
            logger.error(f"Error creating IAM role: {e}")
            raise
    
    def _add_s3_data_source(self, bucket_name):
        try:
            response = bedrock_agent_client.create_data_source(
                knowledgeBaseId=self.kb_id,
                name=f"s3-source-{bucket_name}",
                dataSourceConfiguration={
                    'type': 'S3',
                    's3Configuration': {
                        'bucketArn': f"arn:aws:s3:::{bucket_name}",
                        'inclusionPrefixes': ['']
                    }
                },
                vectorIngestionConfiguration={
                    'chunkingConfiguration': {
                        'chunkingStrategy': self.chunking_strategy
                    }
                }
            )
            logger.info(f"Added S3 data source: {bucket_name}")
            return response['dataSource']['dataSourceId']
            
        except Exception as e:
            logger.error(f"Error adding data source: {e}")
            raise
    
    def get_knowledge_base_id(self):
        return self.kb_id
    
    def start_ingestion_job(self):
        try:
            response = bedrock_agent_client.start_ingestion_job(
                knowledgeBaseId=self.kb_id
            )
            logger.info(f"Started ingestion job: {response['ingestionJob']['ingestionJobId']}")
            return response['ingestionJob']['ingestionJobId']
            
        except Exception as e:
            logger.error(f"Error starting ingestion job: {e}")
            raise
    
    def delete_kb(self, delete_s3_bucket=False):
        try:
            if self.kb_id:
                # Delete data sources first
                data_sources = bedrock_agent_client.list_data_sources(knowledgeBaseId=self.kb_id)
                for ds in data_sources.get('dataSourceSummaries', []):
                    bedrock_agent_client.delete_data_source(
                        knowledgeBaseId=self.kb_id,
                        dataSourceId=ds['dataSourceId']
                    )
                    logger.info(f"Deleted data source: {ds['dataSourceId']}")
                
                # Delete knowledge base
                bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=self.kb_id)
                logger.info(f"Deleted Knowledge Base: {self.kb_id}")
                
                # Delete S3 bucket if requested
                if delete_s3_bucket:
                    for source in self.data_sources:
                        if source["type"] == "S3":
                            # Delete all objects in the bucket
                            objects = s3_client.list_objects_v2(Bucket=source["bucket_name"])
                            if 'Contents' in objects:
                                delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects['Contents']]}
                                s3_client.delete_objects(Bucket=source["bucket_name"], Delete=delete_keys)
                            
                            # Delete the bucket
                            s3_client.delete_bucket(Bucket=source["bucket_name"])
                            logger.info(f"Deleted S3 bucket: {source['bucket_name']}")
                
        except Exception as e:
            logger.error(f"Error deleting resources: {e}")
            raise

# Initialize KB
def initialize_kb():
    global kb_instance
    
    # Define data sources
    data_sources = [
        {"type": "S3", "bucket_name": data_bucket_name}
    ]
    
    # Create knowledge base
    kb_instance = BedrockKnowledgeBase(
        kb_name=knowledge_base_name,
        kb_description="Knowledge base for PDF documents",
        data_sources=data_sources,
        chunking_strategy="FIXED_SIZE", 
        suffix=suffix
    )
    
    return kb_instance.get_knowledge_base_id()

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Upload to S3
                s3_client.upload_file(file_path, data_bucket_name, filename)
                
                # Start ingestion
                ingestion_job_id = kb_instance.start_ingestion_job()
                
                return jsonify({
                    'message': 'File uploaded successfully',
                    'filename': filename,
                    'ingestion_job_id': ingestion_job_id
                })
                
            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                return jsonify({'error': str(e)}), 500
            
            finally:
                # Clean up local file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('upload.html')

@app.route('/query', methods=['GET', 'POST'])
def query_kb():
    if request.method == 'POST':
        query = request.form.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        try:
            # Get KB ID
            kb_id = kb_instance.get_knowledge_base_id()
            
            # Query the KB
            response = bedrock_agent_runtime_client.retrieve(
                    retrievalQuery= {
                        'text': query
                    },
                    knowledgeBaseId=kb_id,
                    retrievalConfiguration= {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 5,
                            'overrideSearchType': "HYBRID", # optional
                        }
                    }
                )
            
            
            retrievalResults = response['retrievalResults']
            contexts = []
            for retrievedResult in retrievalResults: 
                contexts.append(retrievedResult['content']['text'])
            
            prompt = f"""
            Human: You are a financial advisor AI system, and provides answers to questions by using fact based and statistical information when possible. 
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            <context>
            {contexts}
            </context>

            <question>
            {query}
            </question>

            The response should be specific and use statistics or numbers when possible.

            Assistant:"""
            messages=[{ "role":'user', "content":[{'type':'text','text': prompt.format(contexts, query)}]}]
            sonnet_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": messages,
                "temperature": 0.5,
                "top_p": 1
                    }  )
            modelId = 'anthropic.claude-3-sonnet-20240229-v1:0' # change this to use a different version from the model provider
            accept = 'application/json'
            contentType = 'application/json'
            response = bedrock_agent_runtime_client.invoke_model(body=sonnet_payload, modelId=modelId, accept=accept, contentType=contentType)
            response_body = json.loads(response.get('body').read())
            response_text = response_body.get('content')[0]['text']
            return response_text
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('query.html')

@app.route('/status')
def status():
    try:
        kb_id = kb_instance.get_knowledge_base_id()
        kb_status = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
        
        # Get ingestion jobs
        ingestion_jobs = bedrock_agent_client.list_ingestion_jobs(knowledgeBaseId=kb_id)
        
        return jsonify({
            'kb_id': kb_id,
            'kb_status': kb_status['knowledgeBase']['status'],
            'ingestion_jobs': [{
                'id': job['ingestionJobId'],
                'status': job['status']
            } for job in ingestion_jobs.get('ingestionJobSummaries', [])]
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    try:
        kb_instance.delete_kb(delete_s3_bucket=True)
        return jsonify({'message': 'Resources cleaned up successfully'})
    except Exception as e:
        logger.error(f"Error cleaning up: {e}")
        return jsonify({'error': str(e)}), 500

# HTML Templates
@app.route('/templates/index.html')
def get_index_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AWS Bedrock Knowledge Base</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1>AWS Bedrock Knowledge Base Demo</h1>
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Upload PDF</h5>
                            <p class="card-text">Upload PDF documents to the knowledge base.</p>
                            <a href="/upload" class="btn btn-primary">Go to Upload</a>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Query Knowledge Base</h5>
                            <p class="card-text">Ask questions about uploaded documents.</p>
                            <a href="/query" class="btn btn-primary">Go to Query</a>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Knowledge Base Status</h5>
                            <div id="status-container">Loading...</div>
                            <button id="refresh-btn" class="btn btn-secondary mt-3">Refresh Status</button>
                            <a href="/cleanup" class="btn btn-danger mt-3 float-end" onclick="return confirm('Are you sure you want to delete all resources?');">Cleanup Resources</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
        function refreshStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('status-container').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    } else {
                        let html = `<p><strong>KB ID:</strong> ${data.kb_id}</p>`;
                        html += `<p><strong>Status:</strong> ${data.kb_status}</p>`;
                        html += '<h6>Ingestion Jobs:</h6>';
                        html += '<ul>';
                        data.ingestion_jobs.forEach(job => {
                            html += `<li>Job ${job.id}: ${job.status}</li>`;
                        });
                        html += '</ul>';
                        document.getElementById('status-container').innerHTML = html;
                    }
                })
                .catch(error => {
                    document.getElementById('status-container').innerHTML = `<div class="alert alert-danger">Error fetching status: ${error}</div>`;
                });
        }

        document.addEventListener('DOMContentLoaded', function() {
            refreshStatus();
            document.getElementById('refresh-btn').addEventListener('click', refreshStatus);
        });
        </script>
    </body>
    </html>
    """

@app.route('/templates/upload.html')
def get_upload_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload PDF</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1>Upload PDF to Knowledge Base</h1>
            <div class="card mt-4">
                <div class="card-body">
                    <form id="upload-form" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="file" class="form-label">Select PDF File</label>
                            <input type="file" class="form-control" id="file" name="file" accept=".pdf" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Upload</button>
                        <a href="/" class="btn btn-secondary">Back to Home</a>
                    </form>
                    <div id="result-container" class="mt-3"></div>
                    <div id="progress-container" class="mt-3 d-none">
                        <div class="progress">
                            <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('upload-form');
            const resultContainer = document.getElementById('result-container');
            const progressContainer = document.getElementById('progress-container');
            const progressBar = document.getElementById('progress-bar');
            
            form.addEventListener('submit', function(event) {
                event.preventDefault();
                
                const formData = new FormData(form);
                resultContainer.innerHTML = '';
                progressContainer.classList.remove('d-none');
                progressBar.style.width = '0%';
                
                // Simulate progress
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += 5;
                    if (progress <= 90) {
                        progressBar.style.width = progress + '%';
                    }
                }, 300);
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    clearInterval(progressInterval);
                    progressBar.style.width = '100%';
                    
                    setTimeout(() => {
                        progressContainer.classList.add('d-none');
                        
                        if (data.error) {
                            resultContainer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        } else {
                            resultContainer.innerHTML = `
                                <div class="alert alert-success">
                                    File "${data.filename}" uploaded successfully!<br>
                                    Ingestion job started with ID: ${data.ingestion_job_id}<br>
                                    <small>It may take a few minutes for the file to be processed.</small>
                                </div>
                            `;
                            form.reset();
                        }
                    }, 500);
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    progressContainer.classList.add('d-none');
                    resultContainer.innerHTML = `<div class="alert alert-danger">Error: ${error}</div>`;
                });
            });
        });
        </script>
    </body>
    </html>
    """

@app.route('/templates/query.html')
def get_query_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Query Knowledge Base</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1>Query Knowledge Base</h1>
            <div class="card mt-4">
                <div class="card-body">
                    <form id="query-form">
                        <div class="mb-3">
                            <label for="query" class="form-label">Enter your question</label>
                            <textarea class="form-control" id="query" name="query" rows="3" required></textarea>
                        </div>
                        <button type="submit" class="btn btn-primary">Submit Query</button>
                        <a href="/" class="btn btn-secondary">Back to Home</a>
                    </form>
                    <div id="loading" class="mt-3 d-none">
                        <div class="d-flex align-items-center">
                            <div class="spinner-border text-primary me-2" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <span>Processing query...</span>
                        </div>
                    </div>
                    <div id="result-container" class="mt-3">
                        <div id="answer-box" class="d-none">
                            <h4>Answer:</h4>
                            <div id="answer" class="p-3 bg-light rounded"></div>
                        </div>
                        <div id="sources-box" class="mt-4 d-none">
                            <h4>Sources:</h4>
                            <div id="sources"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('query-form');
            const loading = document.getElementById('loading');
            const answerBox = document.getElementById('answer-box');
            const answer = document.getElementById('answer');
            const sourcesBox = document.getElementById('sources-box');
            const sources = document.getElementById('sources');
            
            form.addEventListener('submit', function(event) {
                event.preventDefault();
                
                const formData = new FormData(form);
                
                // Show loading
                loading.classList.remove('d-none');
                answerBox.classList.add('d-none');
                sourcesBox.classList.add('d-none');
                
                fetch('/query', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.classList.add('d-none');
                    
                    if (data.error) {
                        answer.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        answerBox.classList.remove('d-none');
                    } else {
                        // Display answer
                        answer.innerHTML = data.answer.replace(/\\n/g, '<br>');
                        answerBox.classList.remove('d-none');
                        
                        // Display retrieved chunks if available
                        if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
                            let sourcesHtml = '';
                            data.retrieved_chunks.forEach((chunk, index) => {
                                sourcesHtml += `
                                    <div class="card mb-3">
                                        <div class="card-header">Source ${index + 1}</div>
                                        <div class="card-body">
                                            <p>${chunk.text}</p>
                                            <small class="text-muted">Location: ${chunk.location || 'N/A'}</small>
                                        </div>
                                    </div>
                                `;
                            });
                            sources.innerHTML = sourcesHtml;
                            sourcesBox.classList.remove('d-none');
                        }
                    }
                })
                .catch(error => {
                    loading.classList.add('d-none');
                    answer.innerHTML = `<div class="alert alert-danger">Error: ${error}</div>`;
                    answerBox.classList.remove('d-none');
                });
            });
        });
        </script>
    </body>
    </html>
    """

# Helper function to render templates
def render_template(template_name):
    if template_name == 'index.html':
        return get_index_template()
    elif template_name == 'upload.html':
        return get_upload_template()
    elif template_name == 'query.html':
        return get_query_template()

if __name__ == '__main__':
    # Initialize the knowledge base
    kb_id = initialize_kb()
    print(f"Knowledge Base initialized with ID: {kb_id}")
    
    # Run the Flask app
    app.run(debug=True, port=5000)

