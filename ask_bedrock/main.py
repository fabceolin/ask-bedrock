import atexit
import json
import logging
import os
import sys
from collections.abc import Callable

import boto3
import click
import yaml
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_aws import ChatBedrock as BedrockChat

# Import prompt_poet Template class correctly
from prompt_poet.template import Template

logger = logging.getLogger(__name__)

if not logger.handlers:
    logger.addHandler(handler := logging.StreamHandler())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.propagate = False

# Add this after the logger initialization
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

logging.basicConfig(level=logging.INFO)

config_file_path = os.path.join(
    os.path.expanduser("~"), ".config", "ask-bedrock", "config.yaml"
)

atexit.register(
    lambda: logger.debug(
        "\nThank you for using Ask Amazon Bedrock! Consider sharing your feedback here: https://pulse.aws/survey/GTRWNHT1"
    )
)


@click.group()
def cli():
    pass


def log_error(msg: str, e: Exception = None):
    logger.error(click.style(msg, fg="red"))
    if e:
        logger.debug(e, exc_info=True)
        logger.error(click.style(e, fg="red"))


@cli.command()
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def converse(context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    start_conversation(config)


@cli.command()
@click.argument("input", required=False)
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
@click.option("-t", "--template", type=click.Path(exists=True), help="Path to prompt template file")
@click.option("-d", "--data", type=click.Path(exists=True), help="Path to template data file (JSON or YAML)")
@click.option("-p", "--preset", help="Name of preset template-data pair to use")
def prompt(input: str, context: str, debug: bool, template: str, data: str, preset: str):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    # If input is None, use empty string
    if input is None:
        input = ""

    # Check if input is valid JSON (for using input as data)
    input_json_data = None
    if input and input.strip().startswith('{') and input.strip().endswith('}'):
        try:
            input_json_data = json.loads(input)
            logger.debug(f"Input appears to be JSON data: {input_json_data}")
            # Only treat JSON input as template data if a template or preset is specified
            if not (template or preset):
                logger.debug("No template or preset specified, using JSON input as direct input to model")
                input_json_data = None  # Don't treat as template data
        except json.JSONDecodeError:
            # Not valid JSON, will use as normal input
            logger.debug("Input looks like JSON but could not be parsed as valid JSON")
            pass
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    # Handle preset selection
    if preset:
        # Define the presets directory
        presets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
        if not os.path.exists(presets_dir):
            os.makedirs(presets_dir)
            logger.warning(f"Presets directory created at {presets_dir}")

        # Check for the preset directory
        preset_dir = os.path.join(presets_dir, preset)
        if not os.path.exists(preset_dir):
            log_error(f"Preset directory not found: {preset_dir}")
            return

        # Look for template file (template.txt)
        preset_template = os.path.join(preset_dir, "template.txt")

        # Check if custom data file is provided with -d
        if not data:
            # No custom data provided, look for preset data
            if not os.path.exists(preset_template):
                log_error(f"Template file not found in preset directory: {preset_template}")
                return

            # Only require preset data file if we don't have JSON input or custom data
            if not input_json_data:
                # Look for data file (data.json or data.yaml)
                preset_data_json = os.path.join(preset_dir, "data.json")
                preset_data_yaml = os.path.join(preset_dir, "data.yaml")
                preset_data = None

                if os.path.exists(preset_data_json):
                    preset_data = preset_data_json
                elif os.path.exists(preset_data_yaml):
                    preset_data = preset_data_yaml
                else:
                    log_error(f"Data file not found in preset directory: {preset_dir}")
                    return

                # Override template and data with preset values
                template = preset_template
                data = preset_data
            else:
                # If we have JSON input, we only need the template
                template = preset_template
                data = None
        else:
            # Custom data is provided with -d, check if the template exists
            if not os.path.exists(preset_template):
                log_error(f"Template file not found in preset directory: {preset_template}")
                return

            # Use preset template with custom data file
            template = preset_template
            # Keep the data file path from command line
            # We already have the data variable set from the command line
            logger.info(f"Using preset template '{preset}' with custom data file: {data}")

        logger.info(f"Using preset template: {template}")

    # Make sure we have both template and data files if one is specified (except with JSON input)
    if (template and not data) and not input_json_data:
        log_error("Both template and data files must be specified together")
        return

    # When data is provided without template, use data file as input
    if data and not template and not preset:
        logger.info(f"Using data file as input: {data}")
        try:
            with open(data, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if data.endswith('.json'):
                    try:
                        parsed_json = json.loads(file_content)
                        input = json.dumps(parsed_json)  # Re-serialize to ensure proper format
                        logger.debug(f"Successfully parsed JSON input from file")
                    except json.JSONDecodeError as e:
                        log_error(f"Invalid JSON in data file: {e}", e)
                        return
                elif data.endswith(('.yaml', '.yml')):
                    try:
                        parsed_yaml = yaml.safe_load(file_content)
                        input = yaml.dump(parsed_yaml)
                        logger.debug(f"Successfully parsed YAML input from file")
                    except yaml.YAMLError as e:
                        log_error(f"Invalid YAML in data file: {e}", e)
                        return
                else:
                    input = file_content
                    logger.debug(f"Using raw file content as input")
        except Exception as e:
            log_error(f"Error reading data file: {e}", e)
            return

    try:
        llm = model_from_config(config)
    except Exception as e:
        log_error("Error while building Bedrock model", e)
        return

    try:
        # Process template and data if provided
        if template:
            # Use prompt-poet Template for rendering
            try:
                # We need to decide whether to use raw_template OR template_path, not both
                # Let's use template_path since that might work better with includes
                template_obj = Template(
                    template_path=template,  # Just use the template file path
                    from_cache=False,
                    from_examples=False
                )

                # Load template data from file or input JSON
                template_data = {}

                # Check if input is JSON (highest priority)
                if input_json_data:
                    template_data = input_json_data
                    logger.debug(f"Using template data from input JSON: {json.dumps(input_json_data)}")
                    # Clear input since we're using it as data
                    input = ""
                # Otherwise load from data file
                elif data:
                    with open(data, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        try:
                            if data.endswith('.json'):
                                template_data = json.loads(file_content)
                            elif data.endswith(('.yaml', '.yml')):
                                template_data = yaml.safe_load(file_content)
                            else:
                                template_data = {"text": file_content}
                        except Exception as e:
                            log_error(f"Error parsing data file: {e}", e)
                            return
                    logger.info(f"Using template data from file: {data}")

                # Log the template before rendering
                with open(template, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                    logger.debug(f"Template before rendering:\n{template_content}")

                rendered_input = template_obj.render_template(template_data)
                logger.debug(f"Template after rendering with data:\n{rendered_input}")
                input = rendered_input
                logger.info(f"Template rendered successfully")
            except Exception as e:
                log_error(f"Error rendering template: {e}", e)
                return

        response = llm.invoke(input=input)
    except Exception as e:
        log_error("Error while generating response", e)
        return

    if not llm.streaming:
        click.secho(response, fg="yellow")


@cli.command()
@click.argument("name")
@click.option("--template", type=click.Path(exists=True), required=True, help="Path to prompt template file")
@click.option("--data", type=click.Path(exists=True), required=True, help="Path to template data file (JSON or YAML)")
def save_preset(name: str, template: str, data: str):
    """Save a template-data pair as a preset for easier reuse"""
    # Define the presets directory structure
    presets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    if not os.path.exists(presets_dir):
        os.makedirs(presets_dir)

    # Create directory for this preset
    preset_dir = os.path.join(presets_dir, name)
    if not os.path.exists(preset_dir):
        os.makedirs(preset_dir)

    # Target paths for the preset files
    preset_template = os.path.join(preset_dir, "template.txt")

    # Determine data file extension
    data_ext = "json" if data.endswith(".json") else "yaml"
    preset_data = os.path.join(preset_dir, f"data.{data_ext}")

    # Copy the files
    import shutil
    shutil.copy2(template, preset_template)
    shutil.copy2(data, preset_data)

    logger.info(f"Preset '{name}' saved successfully")
    logger.info(f"Template: {preset_template}")
    logger.info(f"Data: {preset_data}")


@cli.command()
def list_presets():
    """List all available presets"""
    presets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
    if not os.path.exists(presets_dir):
        logger.info("No presets directory found. Use 'save_preset' to create presets.")
        return

    # Find all preset directories
    preset_dirs = [d for d in os.listdir(presets_dir)
                 if os.path.isdir(os.path.join(presets_dir, d))]

    if not preset_dirs:
        logger.info("No presets found")
        return

    logger.info("Available presets:")
    for preset_name in sorted(preset_dirs):
        preset_dir = os.path.join(presets_dir, preset_name)
        template_path = os.path.join(preset_dir, "template.txt")
        data_json_path = os.path.join(preset_dir, "data.json")
        data_yaml_path = os.path.join(preset_dir, "data.yaml")

        has_template = os.path.exists(template_path)
        has_data = os.path.exists(data_json_path) or os.path.exists(data_yaml_path)

        if has_template and has_data:
            logger.info(f"  - {preset_name}")
        else:
            missing = []
            if not has_template:
                missing.append("template.txt")
            if not has_data:
                missing.append("data.json/yaml")
            logger.warning(f"  - {preset_name} (incomplete preset, missing: {', '.join(missing)})")


@cli.command()
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def configure(context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    existing_config = get_config(context)
    config = create_config(existing_config)
    if config is not None:
        put_config(context, config)


def start_conversation(config: dict):
    try:
        llm = model_from_config(config)
    except Exception as e:
        log_error("Error while building Bedrock model", e)
        return

    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="Assistant"),
    )

    while True:
        prompt = multiline_prompt(
            lambda: click.prompt(click.style(">>>", fg="green")), return_newlines=True
        )

        try:
            response = conversation.predict(input=prompt)
        except Exception as e:
            log_error("Error while generating response", e)
            continue

        if not llm.streaming:
            click.secho(response, fg="yellow")


def get_config(context: str) -> dict:
    if not os.path.exists(config_file_path):
        return None
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not "contexts" in config:
        return None
    if not context in config["contexts"]:
        return None
    return config["contexts"][context]


# Stores a config for a given context physically
def put_config(context: str, new_config: dict):
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            current_config_file = yaml.safe_load(f)
    else:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        current_config_file = {"contexts": {}}
    new_contexts = current_config_file["contexts"] | {context: new_config}
    new_config_file = current_config_file | {"contexts": new_contexts}

    with open(config_file_path, "w", encoding="utf-8") as f:
        logger.info(f"Writing configuration to {config_file_path}.")
        f.write(yaml.dump(new_config_file))


# Leads through a new configuration dialog
def create_config(existing_config: str) -> dict:
    available_profiles = click.Choice(boto3.session.Session().available_profiles)
    if len(available_profiles.choices) == 0:
        log_error(
            "No profiles found. Make sure you have configured the AWS CLI with at least one profile."
        )
        return None
    aws_profile = click.prompt(
        "👤 AWS profile",
        type=available_profiles,
        default=existing_config["aws_profile"] if existing_config else None,
    )
    region = click.prompt(
        "🌍 Bedrock region",
        default=existing_config["region"] if existing_config else None,
    )

    bedrock = boto3.Session(profile_name=aws_profile).client("bedrock", region)

    try:
        all_models = bedrock.list_foundation_models()["modelSummaries"]
    except Exception as e:
        log_error("Error listing foundation models", e)
        return None

    applicable_models = [
        model
        for model in all_models
        if model["outputModalities"] == ["TEXT"]
        and "TEXT" in model["inputModalities"]  # multi-modal input models are allowed
        and "ON_DEMAND" in model["inferenceTypesSupported"]
    ]

    available_models = click.Choice([model["modelId"] for model in applicable_models])
    model_id = click.prompt(
        "🚗 Model",
        type=available_models,
        default=existing_config["model_id"] if existing_config else None,
    )

    model_params = multiline_prompt(
        lambda: click.prompt(
            "🔠 Model params (JSON)",
            default=existing_config["model_params"] if existing_config else "{}",
        ),
        return_newlines=False,
    )
    config = {
        "region": region,
        "aws_profile": aws_profile,
        "model_id": model_id,
        "model_params": model_params,
    }

    llm = model_from_config(config)
    prompt = "Human: You are an assistant used in a CLI tool called 'Ask Bedrock'. The user has just completed their configuration. Write them a nice hello message, including saying that it is from you.\nAssistant:"

    try:
        response = llm.invoke(prompt)
        if not llm.streaming:
            click.secho(response, fg="yellow")
    except Exception as e:
        if isinstance(e, ValueError) and "AccessDeniedException" in str(e):
            click.secho(
                f"{e}\nAccess denied while trying out the model. Have you enabled model access? Go to the Amazon Bedrock console and select 'Model access' to make sure. Alternatively, choose a different model.",
                fg="red",
            )
            return None
        else:
            click.secho(
                f"{e}\nSomething went wrong while trying out the model, not saving this.",
                fg="red",
            )
            return None

    return config


class YellowStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(click.style(token, fg="yellow"))
        sys.stdout.flush()

    def on_llm_end(self, response, **kwargs) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


# Tries to find a config, creates one otherwise
def init_config(context: str) -> dict:
    config = get_config(context)
    if not config:
        click.echo(
            f"No configuration found for context {context}. Creating new configuration."
        )
        config = create_config(None)
        put_config(context, config)

    if config:
        config = migrate_claude_api(context, config)

    return config


def model_from_config(config: dict) -> BedrockChat:
    model_id = config["model_id"]
    credentials_profile_name = config["aws_profile"]
    region = config["region"]
    bedrock = boto3.Session(profile_name=credentials_profile_name).client(
        "bedrock", region
    )
    streaming = bedrock.get_foundation_model(modelIdentifier=model_id)["modelDetails"][
        "responseStreamingSupported"
    ]

    return BedrockChat(
        credentials_profile_name=credentials_profile_name,
        model_id=model_id,
        region_name=region,
        streaming=streaming,
        callbacks=[YellowStreamingCallbackHandler()],
        model_kwargs=json.loads(config["model_params"]),
    )


def multiline_prompt(prompt: Callable[[], str], return_newlines: bool) -> str:
    response = prompt()
    if response.startswith("<<<"):
        response = response[3:]
        newlines = "\n" if return_newlines else ""
        while not response.endswith(">>>"):
            response += newlines + prompt()
        response = response[:-3]
    return response


def migrate_claude_api(context: str, config: dict):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    if "max_tokens_to_sample" in config["model_params"]:
        logger.info(
            f"Old Claude configuration ('max_tokens_to_sample') found. Migrating to the new version."
        )
        model_params = json.loads(config["model_params"])
        model_params["max_tokens"] = model_params.pop("max_tokens_to_sample")
        config["model_params"] = json.dumps(model_params)
        put_config(context, config)
    return config


if __name__ == "__main__":
    cli()
