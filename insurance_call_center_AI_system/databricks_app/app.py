from mlflow.deployments import get_deploy_client
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config
import gradio as gr
import os
from gradio.themes.utils import sizes
from databricks.sdk import WorkspaceClient
from datetime import datetime, timedelta, timezone
import os
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
    cfg = Config() # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
        
def filter_customer_profile(df, start_time, end_time, phone_number):
    start_time = datetime.fromtimestamp(start_time)
    end_time = datetime.fromtimestamp(end_time)
    # print(f"start_time: {start_time}, end_time: {end_time}, phone_number: {phone_number}")
    df_filtered = df[(df["call_timestamp"] >= start_time) & (df["call_timestamp"] <= end_time)]
    if phone_number not in df_filtered['phone_number'].to_list():
        return "Invalid selection"
    df_filtered = df_filtered[(df_filtered["phone_number"] == phone_number)]
    return [df_filtered.loc[df_filtered['phone_number'] == phone_number, 'customer_name'].values[0], 
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'customer_tenancy'].values[0], 
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'email'].values[0],
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'policy_number'].values[0],
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'automobile'].values[0]]

def process_input(user_input):
    # Process the user input if needed
    return f"Write a email response to caller {user_input} base on the last call transcript?"

def respond_cs_agent(message, history=False):
    endpoint = os.getenv('SERVING_ENDPOINT-cs-agent')
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

theme = gr.themes.Soft(
    text_size=sizes.text_sm,
    radius_size=sizes.radius_sm,
    spacing_size=sizes.spacing_sm,
)

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
df['call_timestamp'] = df['call_timestamp'].apply(lambda x: x.tz_localize(None).to_pydatetime())

with gr.Blocks() as app:
    with gr.Tabs(selected=0) as main_tabs:
        # Main Menu Tab
        with gr.Tab(label="Main Menu", id=0):
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="display:flex; justify-content:center; margin-bottom:10px">
                        <span style="font-size:80px">🤖</span>
                    </div>
                """)
            gr.Markdown("""
                ## Welcome to the Insurance Operator AI Assistant!
                
                I'm here to help you with Insurance Operations.
                
                ### How to use this tool:
                * Select one of the options below to begin
                * Follow the instructions on the next screen
                * Return to this menu anytime using the back button
            """)
            with gr.Row():
                btn1 = gr.Button("Customer Response", variant="primary")
                btn2 = gr.Button("Policy Lookup", variant="primary")
        
        # Improve Customer Happiness Tab
        with gr.Tab(label="Customer Response", id=1):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Here are the Unhappy Customers: select by phone number")
                    start_date = gr.DateTime(label="Start Date", value='2024-01-01 00:00:00')
                    end_date = gr.DateTime(label="End Date", value='2024-12-31 00:00:00')
                    phone_number_selector = gr.Dropdown(list(df['phone_number'].values), label="Phone Number")
                    customer_name_output = gr.Textbox(label="Customer Name", placeholder="Customer Name")
                    tenancy_output = gr.Textbox(label="Customer Tenancy", placeholder="Customer Tenancy")
                    email_output = gr.Textbox(label="Email", placeholder="Email")
                    policy_number_output = gr.Textbox(label="Policy Number", placeholder="Policy Number")
                    auto_output = gr.Textbox(label="Automobile", placeholder="Automobile")
                    @gr.on(inputs=[start_date, end_date, phone_number_selector], outputs=[customer_name_output, tenancy_output, email_output, policy_number_output, auto_output])
                    def update_filter(start_date, end_date, phone_number_selector):
                        return filter_customer_profile(df, start_date, end_date, phone_number_selector)
                    process_btn = gr.Button("Process Current Customer")
                    output_box_process_btn = gr.Textbox(label="Write a Response")

                process_btn.click(
                    process_input, 
                    inputs=[phone_number_selector], 
                    outputs=[output_box_process_btn]
                )
            
                with gr.Column(scale=3):
                    gr.Markdown("# Databricks App - Customer Service CRM Assistant")
                    gr.Markdown("## Instruction")
                    gr.Markdown("1. Filter the start and end date for the calls")
                    gr.Markdown("1. Select a customer from the Phone Number dropdown list")
                    gr.Markdown("2. Click on the Process Current Customer button")
                    gr.Markdown("3. Click on the Submit button")
                    chat_interface = gr.ChatInterface(
                        respond_cs_agent,
                        chatbot=gr.Chatbot(
                            show_label=False, container=False, show_copy_button=True, bubble_full_width=True, height=500,
                        ),
                        textbox=gr.Textbox(placeholder="Draft an email response ...", container=False, scale=0),
                        cache_examples=False,
                        theme=theme,
                        additional_inputs_accordion="Settings",
                    )

                # Connect the processed output to the ChatInterface
                output_box_process_btn.change(
                    lambda x: x,  # Identity function to pass the value unchanged
                    inputs=[output_box_process_btn],
                    outputs=[chat_interface.textbox]  # Access the textbox of ChatInterface
                )
            back_btn1 = gr.Button("⬅ Back to Main Menu")
        # Answer Policy Questions Tab
        with gr.Tab(label="Answer Policy Questions", id=2):
            chat_interface = gr.ChatInterface(
                respond_policy_doc_rag,
                chatbot=gr.Chatbot(
                    show_label=False, container=False, show_copy_button=True, bubble_full_width=True, height=500,
                ),
                textbox=gr.Textbox(placeholder="What is our home accident claim policy?", container=False, scale=0),
                cache_examples=False,
                theme=theme,
                additional_inputs_accordion="Settings",
            )
            back_btn2 = gr.Button("⬅ Back to Main Menu")

 # Navigation Logic
    btn1.click(lambda: gr.Tabs(selected=1), None, main_tabs)
    btn2.click(lambda: gr.Tabs(selected=2), None, main_tabs)
    back_btn1.click(lambda: gr.Tabs(selected=0), None, main_tabs)
    back_btn2.click(lambda: gr.Tabs(selected=0), None, main_tabs)


if __name__ == "__main__":
    app.launch()
