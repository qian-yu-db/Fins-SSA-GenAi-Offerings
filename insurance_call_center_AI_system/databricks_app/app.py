from mlflow.deployments import get_deploy_client
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config
import gradio as gr
import os
from gradio.themes.utils import sizes
from databricks.sdk import WorkspaceClient
from datetime import datetime, timedelta, timezone
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
client = get_deploy_client("databricks")

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT-cs-agent'), "SERVING_ENDPOINT-cs-agent must be set in app.yaml."
assert os.getenv('SERVING_ENDPOINT-doc-rag'), "SERVING_ENDPOINT-doc-rag must be set in app.yaml."
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."


def sqlQuery(query: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a pandas DataFrame with robust error handling"""
    try:
        cfg = Config() # Pull environment variables for auth

        # Verify warehouse ID is set
        warehouse_id = os.getenv('DATABRICKS_WAREHOUSE_ID')
        if not warehouse_id:
            logger.error("DATABRICKS_WAREHOUSE_ID environment variable not set")
            return pd.DataFrame()  # Return empty dataframe

        # Log query for debugging (truncate if too long)
        truncated_query = query[:200] + "..." if len(query) > 200 else query
        logger.info(f"Executing SQL query: {truncated_query}")

        # Execute the query
        with sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall_arrow().to_pandas()

                # Verify result is a DataFrame
                if not isinstance(result, pd.DataFrame):
                    logger.error(f"SQL query returned {type(result)} instead of DataFrame")
                    return pd.DataFrame()  # Return empty dataframe

                logger.info(f"Query executed successfully. Result shape: {result.shape}")
                return result

    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        # Return empty dataframe on error
        return pd.DataFrame()
        
def filter_customer_profile(df, start_time, end_time, phone_number):
    # First, verify that df is a valid dataframe and not a scalar value
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid dataframe received: {type(df)}")
        return "Error: Invalid data", "", "", "", ""

    # Check if df is empty
    if df.empty:
        logger.error("Empty dataframe received")
        return "Error: No data available", "", "", "", ""

    # Check if required columns exist
    required_columns = ["call_timestamp", "phone_number", "customer_name", "customer_tenancy", "email", "policy_number", "automobile"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns in dataframe: {missing_columns}")
        return f"Error: Missing data columns", "", "", "", ""

    # Handle different date input formats from Gradio DateTime component
    # Convert to datetime object regardless of input type
    try:
        if isinstance(start_time, str):
            try:
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    start_time = datetime.strptime(start_time, '%Y-%m-%d')
                except ValueError:
                    start_time = datetime(2024, 1, 1)
        elif isinstance(start_time, (int, float)):
            # Convert timestamp to datetime
            start_time = datetime.fromtimestamp(start_time)
        elif not isinstance(start_time, datetime):
            logger.warning(f"Unknown start_time type: {type(start_time)}, using default")
            start_time = datetime(2024, 1, 1)
    except Exception as e:
        logger.error(f"Error converting start_time: {e}")
        start_time = datetime(2024, 1, 1)

    try:
        if isinstance(end_time, str):
            try:
                end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    end_time = datetime.strptime(end_time, '%Y-%m-%d')
                except ValueError:
                    end_time = datetime(2024, 12, 31)
        elif isinstance(end_time, (int, float)):
            # Convert timestamp to datetime
            end_time = datetime.fromtimestamp(end_time)
        elif not isinstance(end_time, datetime):
            logger.warning(f"Unknown end_time type: {type(end_time)}, using default")
            end_time = datetime(2024, 12, 31)
    except Exception as e:
        logger.error(f"Error converting end_time: {e}")
        end_time = datetime(2024, 12, 31)

    # Log values for debugging
    logger.info(f"Filtering with start_time: {start_time}, end_time: {end_time}, phone: {phone_number}")
    logger.info(f"Sample data: {df.head(1).to_dict()}")

    try:
        # Convert call_timestamp column to datetime if needed
        if 'call_timestamp' in df.columns and not df.empty:
            # First check the type of the first timestamp to decide how to handle it
            sample_timestamp = df["call_timestamp"].iloc[0]
            logger.info(f"Sample timestamp type: {type(sample_timestamp)}")

            # Make sure timestamps are datetime objects for comparison
            if not isinstance(sample_timestamp, datetime):
                logger.info("Converting call_timestamp column to datetime")
                try:
                    # For pandas Timestamp objects
                    if hasattr(df["call_timestamp"], 'dt'):
                        df["call_timestamp_dt"] = df["call_timestamp"].dt.to_pydatetime()
                    else:
                        # For other types, attempt conversion
                        df["call_timestamp_dt"] = df["call_timestamp"].apply(
                            lambda x: x if isinstance(x, datetime) else
                                    datetime.fromtimestamp(x) if isinstance(x, (int, float)) else
                                    datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') if isinstance(x, str) else
                                    datetime(2024, 1, 1)  # fallback
                        )
                    # Use the converted column for filtering
                    df_filtered = df[(df["call_timestamp_dt"] >= start_time) & (df["call_timestamp_dt"] <= end_time)]
                except Exception as e:
                    logger.error(f"Error converting timestamps for comparison: {e}")
                    # Fallback to string comparison if datetime conversion fails
                    start_str = start_time.strftime('%Y-%m-%d')
                    end_str = end_time.strftime('%Y-%m-%d')
                    df["call_date_str"] = df["call_timestamp"].astype(str).str[:10]  # Get just the date part
                    df_filtered = df[(df["call_date_str"] >= start_str) & (df["call_date_str"] <= end_str)]
            else:
                # Direct comparison if already datetime objects
                df_filtered = df[(df["call_timestamp"] >= start_time) & (df["call_timestamp"] <= end_time)]
        else:
            logger.warning("call_timestamp column missing or df empty, returning all records")
            df_filtered = df  # Return all records if can't filter

        # Check if we have matching records after date filter
        if df_filtered.empty:
            logger.warning("No records found in date range")
            return "No records in selected date range", "", "", "", ""

        # Check if phone number exists and is valid
        if phone_number is None or phone_number == "":
            logger.warning("No phone number selected")
            return "Please select a phone number", "", "", "", ""

        # Check if phone number exists in filtered results
        if phone_number in df_filtered['phone_number'].values:
            df_customer = df_filtered[df_filtered["phone_number"] == phone_number]
            if not df_customer.empty:
                logger.info(f"df_customer: {df_customer.to_string()}")
                return [
                    df_customer['customer_name'].values[0],
                    df_customer['customer_tenancy'].values[0],
                    df_customer['email'].values[0],
                    df_customer['policy_number'].values[0],
                    df_customer['automobile'].values[0]
                ]
            else:
                logger.warning(f"No customer data found for phone: {phone_number}")
        else:
            logger.warning(f"Phone number {phone_number} not found in filtered data")

        return "No matching customer found", "", "", "", ""

    except Exception as e:
        logger.error(f"Error in filter_customer_profile: {str(e)}")
        return f"Error: {str(e)}", "", "", "", ""

def respond_cs_agent(message):
    endpoint = os.getenv('SERVING_ENDPOINT-cs-agent')
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"
    try:
        messages = [{"role": "user", "content": message}]
        response = client.predict(
            endpoint=endpoint,
            inputs={
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 500,
            },
        )
    except Exception as error:
        return f"ERROR requesting endpoint {endpoint}: {error}"
    return response['messages'][-1]['content']

def respond_policy_doc_rag(message, history=True):
    endpoint = os.getenv('SERVING_ENDPOINT-doc-rag')
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"
    try:
        messages = []
        if history:
            for human, assistant in history:
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})
        response = client.predict(
            endpoint=endpoint,
            inputs={
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 500,
            },
        )
    except Exception as error:
        return f"ERROR requesting endpoint {endpoint}: {error}"
    return response['messages'][-1]['content']

# Load data
def load_data():
    try:
        # Query the database
        df = sqlQuery("""select
                        phone_number,
                        call_timestamp,
                        sentiment,
                        concat(first_name, ' ', last_name) as customer_name,
                        concat('Customer since ', cast(issue_date as string)) as customer_tenancy,
                        email,
                        policy_number,
                        concat(model_year, ' ', make, ' ', model) as automobile
                        from fins_genai.call_center.call_center_transcripts_analysis
                        where sentiment = 'negative' or sentiment = 'mixed'
                        """)

        logger.info(f"query results: {df.head().to_string()}")
        # Verify that we got a dataframe back
        if not isinstance(df, pd.DataFrame):
            logger.error(f"sqlQuery did not return a DataFrame, got {type(df)} instead")
            # Return an empty dataframe with the expected columns
            return pd.DataFrame(columns=[
                "phone_number", "call_timestamp", "sentiment", "customer_name",
                "customer_tenancy", "email", "policy_number", "automobile"
            ])

        # Process timestamp column if it exists
        if 'call_timestamp' in df.columns and not df.empty:
            try:
                df['call_timestamp'] = df['call_timestamp'].apply(
                    lambda x: x.tz_localize(None).to_pydatetime() if x is not None else datetime(2024, 1, 1)
                )
            except Exception as e:
                logger.error(f"Error processing timestamps: {str(e)}")
                # Set all timestamps to a default if processing fails
                df['call_timestamp'] = datetime(2024, 1, 1)
        else:
            # If column is missing, add it with default value
            logger.warning("call_timestamp column missing, adding with default values")
            df['call_timestamp'] = datetime(2024, 1, 1)

        # Log dataframe info for debugging
        logger.info(f"Loaded dataframe with shape: {df.shape}")
        if not df.empty:
            logger.info(f"Column dtypes: {df.dtypes}")

        return df

    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        # Return an empty dataframe with the expected columns on error
        return pd.DataFrame(columns=[
            "phone_number", "call_timestamp", "sentiment", "customer_name",
            "customer_tenancy", "email", "policy_number", "automobile"
        ])

# Theme setup
theme = gr.themes.Soft(
    text_size=sizes.text_sm,
    radius_size=sizes.radius_sm,
    spacing_size=sizes.spacing_sm,
)

# Create the app
with gr.Blocks(theme=theme, title="Insurance Operator AI Assistant") as app:
    # App state variables - create fallback empty dataframe
    try:
        df = load_data()
        # Double-check that df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.error(f"load_data returned {type(df)}, creating empty dataframe")
            df = pd.DataFrame(columns=[
                "phone_number", "call_timestamp", "sentiment", "customer_name",
                "customer_tenancy", "email", "policy_number", "automobile"
            ])
            # Add some dummy data for testing
            df.loc[0] = ["123-456-7890", datetime(2024, 1, 1), "negative",
                        "Test Customer", "Customer since 2020-01-01",
                        "test@example.com", "POL123456", "2022 Test Model"]
    except Exception as e:
        logger.error(f"Error initializing dataframe: {str(e)}")
        df = pd.DataFrame(columns=[
            "phone_number", "call_timestamp", "sentiment", "customer_name",
            "customer_tenancy", "email", "policy_number", "automobile"
        ])
        # Add some dummy data for testing
        df.loc[0] = ["123-456-7890", datetime(2024, 1, 1), "negative",
                    "Test Customer", "Customer since 2020-01-01",
                    "test@example.com", "POL123456", "2022 Test Model"]
    
    # Main Container
    with gr.Column():
        # Header
        with gr.Row():
            gr.HTML("""
                <div style="display:flex; align-items:center; margin-bottom:20px">
                    <span style="font-size:50px; margin-right:15px">🤖</span>
                    <h1 style="margin:0">Insurance Operator AI Assistant</h1>
                </div>
            """)
        
        # Navigation Tabs
        with gr.Tabs() as main_tabs:
            # Home Tab
            with gr.Tab("Home", id=0):
                with gr.Column(scale=1):
                    gr.Markdown("""
                        ## Welcome to the Insurance Operator AI Assistant!
                        
                        I'm here to help you with Insurance Operations.
                        
                        ### Choose an option below:
                    """)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML("""
                                <div style="border:1px solid #ddd; border-radius:10px; padding:20px; margin:10px; text-align:center">
                                    <h3>📝 Customer Response</h3>
                                    <p>Generate responses for customer emails</p>
                                </div>
                            """)
                            response_btn = gr.Button("Open Customer Response", variant="primary")
                        
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML("""
                                <div style="border:1px solid #ddd; border-radius:10px; padding:20px; margin:10px; text-align:center">
                                    <h3>❓ Policy Lookup</h3>
                                    <p>Ask questions about our insurance policies</p>
                                </div>
                            """)
                            policy_btn = gr.Button("Open Policy Lookup", variant="primary")
            
            # Customer Response Tab
            with gr.Tab("Customer Response", id=1):
                with gr.Row():
                    # Customer Selector Panel
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Select Customer")
                        # Use string format that matches Gradio's expected format (with time component)
                        start_date = gr.DateTime(label="Start Date", value="2024-01-01 00:00:00")
                        end_date = gr.DateTime(label="End Date", value="2024-12-31 23:59:59")
                        with gr.Accordion("Customer Information", open=True):
                            # Safely get phone numbers list with fallback
                            try:
                                phone_numbers = list(df['phone_number'].values) if isinstance(df, pd.DataFrame) and 'phone_number' in df.columns else ["123-456-7890"]
                            except Exception as e:
                                logger.error(f"Error getting phone numbers: {str(e)}")
                                phone_numbers = ["123-456-7890"]  # Fallback

                            phone_number_selector = gr.Dropdown(
                                phone_numbers,
                                label="Phone Number",
                                info="Select a customer to generate a response"
                            )
                            customer_name_output = gr.Textbox(label="Customer Name", interactive=False)
                            tenancy_output = gr.Textbox(label="Customer Tenancy", interactive=False)
                            email_output = gr.Textbox(label="Email", interactive=False)
                            policy_number_output = gr.Textbox(label="Policy Number", interactive=False)
                            auto_output = gr.Textbox(label="Automobile", interactive=False)
                        
                        lookup_btn = gr.Button("Lookup Customer", variant="primary")
                        
                        # Update customer info when phone number changes
                        # Wrapper function to pass the dataframe correctly
                        def lookup_customer(start_date, end_date, phone_number):
                            # Access the global dataframe
                            return filter_customer_profile(df, start_date, end_date, phone_number)

                        lookup_btn.click(
                            lookup_customer,
                            inputs=[start_date, end_date, phone_number_selector],
                            outputs=[customer_name_output, tenancy_output, email_output, policy_number_output, auto_output]
                        )
                    
                    # Response Generator Panel
                    with gr.Column(scale=2, min_width=600):
                        gr.Markdown("### Generate Customer Response")
                        
                        # Response prompt
                        prompt_template = gr.Textbox(
                            label="Response Prompt", 
                            value="Write an email response to this customer based on their most recent call transcript.",
                            lines=2
                        )
                        
                        # Generate button
                        generate_btn = gr.Button("Generate Response", variant="primary")
                        
                        # Response area
                        with gr.Accordion("Response", open=True):
                            response_output = gr.Textbox(
                                label="Generated Response",
                                lines=10,
                                max_lines=20,
                                interactive=True,
                                placeholder="Generated response will appear here..."
                            )
                            
                            # Action buttons with copy area in a more organized layout
                            with gr.Column():
                                with gr.Row():
                                    copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary")
                                    regenerate_btn = gr.Button("🔄 Regenerate", variant="secondary")

                                # Create a collapsible section for the copy area
                                with gr.Accordion("Copy Text", open=False):
                                    gr.Markdown("*Select all text (Ctrl+A) then copy (Ctrl+C)*")
                                    copy_area = gr.Textbox(
                                        label=None,
                                        interactive=True,
                                        lines=5,
                                        max_lines=5,
                                        show_copy_button=True  # Use Gradio's built-in copy button
                                    )

                            def copy_to_clipboard(text):
                                return text

                            copy_btn.click(
                                copy_to_clipboard,
                                inputs=response_output,
                                outputs=copy_area
                            )
                        
                        # Generate response when button is clicked
                        def generate_response(phone_number, prompt):
                            if not phone_number:
                                return "Please select a valid customer phone number."
                            
                            customized_prompt = f"{prompt} Customer with phone number {phone_number}"
                            return respond_cs_agent(customized_prompt)
                        
                        generate_btn.click(
                            generate_response,
                            inputs=[phone_number_selector, prompt_template],
                            outputs=[response_output]
                        )
                        
                        # Regenerate button handler
                        regenerate_btn.click(
                            generate_response,
                            inputs=[phone_number_selector, prompt_template],
                            outputs=[response_output]
                        )
                        
                # Back button
                home_btn1 = gr.Button("⬅ Back to Home")
            
            # Policy Q&A Tab
            with gr.Tab("Policy Lookup", id=2):
                gr.Markdown("### Policy Information Lookup")
                gr.Markdown("Ask questions about our insurance policies and coverage details.")
                
                # Q&A Chat Interface
                chat_interface = gr.ChatInterface(
                    respond_policy_doc_rag,
                    chatbot=gr.Chatbot(
                        show_label=False, 
                        container=False, 
                        show_copy_button=True, 
                        bubble_full_width=True, 
                        height=500,
                    ),
                    textbox=gr.Textbox(
                        placeholder="What is our home accident claim policy?", 
                        container=False, 
                        scale=0
                    ),
                    cache_examples=False,
                    theme=theme,
                )
                
                # Back button
                home_btn2 = gr.Button("⬅ Back to Home")
        
        # Navigation Logic
        response_btn.click(lambda: gr.Tabs(selected=1), None, main_tabs)
        policy_btn.click(lambda: gr.Tabs(selected=2), None, main_tabs)
        home_btn1.click(lambda: gr.Tabs(selected=0), None, main_tabs)
        home_btn2.click(lambda: gr.Tabs(selected=0), None, main_tabs)

if __name__ == "__main__":
    app.launch()
