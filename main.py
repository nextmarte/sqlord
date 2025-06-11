import gradio as gr
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent import AgentExecutor
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from io import StringIO

# import dotenv # Descomente se for usar python-dotenv e um arquivo .env
# dotenv.load_dotenv() # Descomente para carregar .env

# Importações adicionais para outros LLMs
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # Define como None se não instalado
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # Define como None se não instalado


# Configurações Iniciais do Banco de Dados e LLM (de variáveis de ambiente como fallback/default)
DEFAULT_DB_USER = os.getenv("POSTGRES_USER")
DEFAULT_DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DEFAULT_DB_HOST = os.getenv("POSTGRES_HOST")
DEFAULT_DB_PORT = os.getenv("POSTGRES_PORT")
DEFAULT_DB_NAME = os.getenv("POSTGRES_DB")

# Chaves de API de variáveis de ambiente
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Usado se DeepSeek não for via OpenAI compat.
DEFAULT_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# DATABASE_URL será definido dinamicamente
# Globais para componentes LangChain e DB
db_engine = None
db = None
llm = None
# db_chain = None # Substituído por agent_executor
agent_executor = None  # Adicionado
chat_memory = None  # Adicionado

# Sistema de tradução - dicionários de idiomas
TRANSLATIONS = {
    "pt-BR": {
        # Interface principal
        "app_title": "SQLord - Converse com seu Banco de Dados PostgreSQL",
        "chat_tab": "Chat com Banco",
        "config_tab": "Configuração",
        "question_placeholder": "Digite sua pergunta em linguagem natural sobre o banco de dados aqui... \nEx: Quantos usuários existem? Quais tabelas possuem a coluna 'nome'?",
        "question_label": "Sua Pergunta",
        "submit_button": "Enviar Pergunta",
        "sql_output_label": "Query SQL Gerada",
        "result_output_label": "Resultado da Query / Resposta do LLM",
        "viz_title": "### Visualização Automática de Dados",
        "viz_description_label": "Sobre esta visualização",

        # Configuração
        "db_config_title": "### Configuração do Banco de Dados PostgreSQL",
        "db_user_label": "Usuário do Banco",
        "db_password_label": "Senha do Banco",
        "db_host_label": "Host do Banco",
        "db_port_label": "Porta do Banco",
        "db_name_label": "Nome do Banco de Dados",
        "llm_config_title": "### Configuração do LLM",
        "llm_provider_label": "Provedor LLM",
        "api_key_label": "Chave API do LLM",
        "api_key_placeholder": "Cole sua chave API aqui (opcional, usa var. de ambiente se vazio)",
        "connect_button": "Aplicar Configurações e Conectar",
        "status_title": "### Status",
        "llm_status_label": "Status do LLM",
        "connection_status_label": "Status da Conexão DB",
        "current_config_label": "Configuração DB Ativa",

        # Mensagens de status
        "no_db_config": "Nenhuma configuração de banco de dados ativa.",
        "waiting_db_config": "Aguardando configuração do banco de dados.",
        "llm_not_configured": "LLM não configurado. Verifique as configurações.",
        "connecting_to": "Tentando conectar a: {user}@{host}:{port}/{db_name}",
        "connected_to": "Conectado a: {user}@{host}:{port}/{db_name}",
        "connection_success": "Conexão com {db_name}@{host} bem-sucedida!",
        "connection_fail": "Falha na conexão com o banco. Verifique os detalhes e tente novamente.",
        "agent_ready": "Agente SQL com memória pronto (kwargs: {kwargs}).",
        "agent_fail": "Agente SQL não pôde ser inicializado: LLM não configurado/disponível. Status LLM: {status}",
        "configuring_llm": "Configurando LLM: {provider}...",
        "llm_success": "LLM ({provider}) inicializado com sucesso.",

        # Visualização
        "line_chart": "Gráfico de Linha",
        "bar_chart": "Gráfico de Barras",
        "histogram": "Histograma",
        "scatter_plot": "Gráfico de Dispersão",
        "pie_chart": "Gráfico de Pizza",
        "legend": "Legenda",
        "evolution_of": "Evolução de {column} ao longo do tempo",
        "by": "{numeric} por {category}",
        "distribution_of": "Distribuição de {column}",
        "relation_between": "Relação entre {column1} e {column2}",
        "count_of": "Contagem de {column}",
        "too_many_columns": "Muitas colunas ({count}) para visualização gráfica eficaz",
        "data_not_suitable": "Dados não adequados para visualização gráfica automática",
        "insufficient_data": "Dados insuficientes para visualização",
        "too_large_dataset": "Conjunto de dados muito grande para visualização ({rows} linhas). Limite: {limit} linhas.",
        "cannot_convert": "Não foi possível converter o resultado em dados tabulares.",
        "no_viz_data": "Sem dados para visualização",

        # Mensagens do chat
        "chat_intro": "Faça uma pergunta em linguagem natural. O sistema (agora com memória) tentará gerar uma query SQL, executá-la e mostrar o resultado.",
        "db_config": "DB Config: {config}",
        "llm_status": "LLM Status: {status}",
        "llm_not_operational": "LLM Status: {status} (Não operacional)",
        "config_alerts": "Alertas de Configuração: {alerts}",
        "db_status": "DB Status: {status}",
        "db_connected": "DB Status: Conectado e pronto.",
        "db_connected_agent_fail": "DB Status: Conectado, mas Agente SQL falhou ao inicializar.",
        "db_not_connected": "DB Status: Não conectado. Configure na aba 'Configuração'.",

        # Mensagens de erros e resultados
        "agent_not_initialized": "O Agente SQL não foi inicializado.",
        "db_not_connected_error": "Banco de dados não conectado.",
        "llm_not_configured_error": "LLM não configurado.",
        "memory_not_configured": "Memória da conversa não configurada.",
        "setup_errors": "Erros de setup LLM: {errors}",
        "llm_status_error": "Status LLM: {status}",
        "db_connection_status": "Status da conexão DB: {status}",
        "check_config": "Verifique a aba 'Configuração' e as variáveis de ambiente.",
        "no_sql_query": "Nenhuma query SQL foi executada por este turno.",
        "no_query_result": "Nenhum resultado de query SQL direto para este turno.",
        "agent_direct_response": "O Agente respondeu diretamente (nenhuma query SQL específica para este turno).",
        "process_error": "Erro ao processar a pergunta com o Agente SQL: {error}",

        # Erros de LLM
        "api_key_error": "Chave API da {provider} não fornecida.",
        "package_error": "Pacote {package} não instalado. Execute: uv pip install {package}",
        "unknown_provider": "Provedor LLM desconhecido: {provider}",
        "import_error": "Erro de importação para {provider}: {error}. Verifique se o pacote está instalado.",
        "init_error": "Erro ao inicializar LLM com {provider}: {error}.",
        "db_error": "Erro ao conectar/configurar o banco de dados: {error}",
        "agent_init_error": "Erro ao inicializar o Agente SQL: {error}",
        "viz_error": "Erro ao criar visualização: {error}",
        "convert_error": "Erro ao converter resultado para DataFrame: {error}"
    },
    "en-US": {
        # Interface principal
        "app_title": "SQLord - Talk to Your PostgreSQL Database",
        "chat_tab": "Chat with Database",
        "config_tab": "Configuration",
        "question_placeholder": "Type your natural language question about the database here... \nEx: How many users exist? Which tables have the 'name' column?",
        "question_label": "Your Question",
        "submit_button": "Submit Question",
        "sql_output_label": "Generated SQL Query",
        "result_output_label": "Query Result / LLM Response",
        "viz_title": "### Automatic Data Visualization",
        "viz_description_label": "About this visualization",

        # Configuração
        "db_config_title": "### PostgreSQL Database Configuration",
        "db_user_label": "Database User",
        "db_password_label": "Database Password",
        "db_host_label": "Database Host",
        "db_port_label": "Database Port",
        "db_name_label": "Database Name",
        "llm_config_title": "### LLM Configuration",
        "llm_provider_label": "LLM Provider",
        "api_key_label": "LLM API Key",
        "api_key_placeholder": "Paste your API key here (optional, uses env variable if empty)",
        "connect_button": "Apply Settings and Connect",
        "status_title": "### Status",
        "llm_status_label": "LLM Status",
        "connection_status_label": "DB Connection Status",
        "current_config_label": "Active DB Configuration",

        # Mensagens de status
        "no_db_config": "No active database configuration.",
        "waiting_db_config": "Waiting for database configuration.",
        "llm_not_configured": "LLM not configured. Check settings.",
        "connecting_to": "Trying to connect to: {user}@{host}:{port}/{db_name}",
        "connected_to": "Connected to: {user}@{host}:{port}/{db_name}",
        "connection_success": "Connection to {db_name}@{host} successful!",
        "connection_fail": "Failed to connect to database. Check details and try again.",
        "agent_ready": "SQL Agent with memory ready (kwargs: {kwargs}).",
        "agent_fail": "SQL Agent could not be initialized: LLM not configured/available. LLM Status: {status}",
        "configuring_llm": "Configuring LLM: {provider}...",
        "llm_success": "LLM ({provider}) initialized successfully.",

        # Visualização
        "line_chart": "Line Chart",
        "bar_chart": "Bar Chart",
        "histogram": "Histogram",
        "scatter_plot": "Scatter Plot",
        "pie_chart": "Pie Chart",
        "legend": "Legend",
        "evolution_of": "Evolution of {column} over time",
        "by": "{numeric} by {category}",
        "distribution_of": "Distribution of {column}",
        "relation_between": "Relationship between {column1} and {column2}",
        "count_of": "Count of {column}",
        "too_many_columns": "Too many columns ({count}) for effective graph visualization",
        "data_not_suitable": "Data not suitable for automatic graphical visualization",
        "insufficient_data": "Insufficient data for visualization",
        "too_large_dataset": "Dataset too large for visualization ({rows} rows). Limit: {limit} rows.",
        "cannot_convert": "Could not convert result to tabular data.",
        "no_viz_data": "No data for visualization",

        # Mensagens do chat
        "chat_intro": "Ask a question in natural language. The system (now with memory) will try to generate an SQL query, execute it, and show the result.",
        "db_config": "DB Config: {config}",
        "llm_status": "LLM Status: {status}",
        "llm_not_operational": "LLM Status: {status} (Not operational)",
        "config_alerts": "Configuration Alerts: {alerts}",
        "db_status": "DB Status: {status}",
        "db_connected": "DB Status: Connected and ready.",
        "db_connected_agent_fail": "DB Status: Connected, but SQL Agent failed to initialize.",
        "db_not_connected": "DB Status: Not connected. Configure in the 'Configuration' tab.",

        # Mensagens de erros e resultados
        "agent_not_initialized": "The SQL Agent was not initialized.",
        "db_not_connected_error": "Database not connected.",
        "llm_not_configured_error": "LLM not configured.",
        "memory_not_configured": "Conversation memory not configured.",
        "setup_errors": "LLM setup errors: {errors}",
        "llm_status_error": "LLM Status: {status}",
        "db_connection_status": "DB connection status: {status}",
        "check_config": "Check the 'Configuration' tab and environment variables.",
        "no_sql_query": "No SQL query was executed for this turn.",
        "no_query_result": "No direct SQL query result for this turn.",
        "agent_direct_response": "The Agent responded directly (no specific SQL query for this turn).",
        "process_error": "Error processing the question with SQL Agent: {error}",

        # Erros de LLM
        "api_key_error": "{provider} API key not provided.",
        "package_error": "Package {package} not installed. Run: uv pip install {package}",
        "unknown_provider": "Unknown LLM provider: {provider}",
        "import_error": "Import error for {provider}: {error}. Check if the package is installed.",
        "init_error": "Error initializing LLM with {provider}: {error}.",
        "db_error": "Error connecting/configuring the database: {error}",
        "agent_init_error": "Error initializing SQL Agent: {error}",
        "viz_error": "Error creating visualization: {error}",
        "convert_error": "Error converting result to DataFrame: {error}"
    }
}

# Configuração global de idioma
DEFAULT_LANGUAGE = os.getenv(
    "SQLORD_LANGUAGE", "pt-BR")  # Português como padrão
current_language = DEFAULT_LANGUAGE


def _(key, **kwargs):
    """
    Função de tradução simples. Retorna o texto traduzido para o idioma atual.
    Se a chave não existir, retorna a própria chave.
    Suporta formatação com kwargs.
    """
    global current_language
    translation = TRANSLATIONS.get(current_language, {}).get(key, key)
    if kwargs:
        try:
            return translation.format(**kwargs)
        except:
            return translation
    return translation


def set_language(language):
    """Altera o idioma atual e retorna mensagem de confirmação"""
    global current_language
    if language in TRANSLATIONS:
        current_language = language
        return _("language_changed", language=language)
    return _("language_not_available", language=language)


# Alterar as variáveis globais para usar o sistema de tradução
current_db_config_display = _("no_db_config")
setup_error_message_global = ""
connection_status_message_global = _("waiting_db_config")
llm_status_message_global = _("llm_not_configured")

LLM_PROVIDERS = [
    "OpenAI", "Anthropic (Claude)", "Google (Gemini)", "DeepSeek (OpenAI compatible)"]


def _apply_postgres_interval_fix(sql_query: str) -> str:
    """
    Corrige a sintaxe de INTERVAL no PostgreSQL de 'INTERVAL N UNIT' para 'INTERVAL 'N UNIT''.
    Exemplo: INTERVAL 1 MONTH -> INTERVAL '1 MONTH'
    Funciona para unidades comuns como YEAR, MONTH, DAY, HOUR, MINUTE, SECOND.
    """
    # Padrão para encontrar "INTERVAL <dígitos> <UNIDADE_MAIUSCULA>" que não está já entre aspas.
    # Esta regex é específica para o padrão problemático observado.
    pattern = r"(INTERVAL\s+)(\d+\s+(?:YEAR|MONTH|DAY|HOUR|MINUTE|SECOND))(\s|;|$)"

    def replacer(match):
        # match.group(1) é "INTERVAL "
        # match.group(2) é "N UNIT" (ex: "1 MONTH")
        # match.group(3) é o caractere de terminação (espaço, ;, ou fim da string)
        return f"{match.group(1)}'{match.group(2)}'{match.group(3)}"

    corrected_sql = re.sub(pattern, replacer, sql_query)
    if corrected_sql != sql_query:
        print(
            f"DEBUG: Aplicada correção de intervalo PostgreSQL na query exibida: De '{sql_query}' Para '{corrected_sql}'")
    return corrected_sql


def _initialize_llm(api_key_from_ui: str, llm_provider: str):
    global llm, setup_error_message_global, llm_status_message_global

    # Prioriza a chave da UI, depois a variável de ambiente específica do provedor, depois a genérica OPENAI_API_KEY para compatibilidade
    api_key = api_key_from_ui  # Chave da UI tem maior prioridade

    llm_status_message_global = _("configuring_llm", provider=llm_provider)
    print(llm_status_message_global)

    try:
        if llm_provider == "OpenAI":
            if not api_key:
                api_key = DEFAULT_OPENAI_API_KEY
            if not api_key:
                raise ValueError(_("api_key_error", provider="OpenAI"))
            llm = ChatOpenAI(temperature=0, openai_api_key=api_key,
                             model_name="gpt-3.5-turbo")
        elif llm_provider == "Anthropic (Claude)":
            if not ChatAnthropic:
                raise ImportError(
                    _("package_error", package="langchain-anthropic"))
            if not api_key:
                api_key = DEFAULT_ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError(_("api_key_error", provider="Anthropic"))
            llm = ChatAnthropic(
                temperature=0, anthropic_api_key=api_key, model_name="claude-3-haiku-20240307")
        elif llm_provider == "Google (Gemini)":
            if not ChatGoogleGenerativeAI:
                raise ImportError(
                    _("package_error", package="langchain-google-genai"))
            if not api_key:
                api_key = DEFAULT_GOOGLE_API_KEY
            if not api_key:
                raise ValueError(_("api_key_error", provider="Google"))
            llm = ChatGoogleGenerativeAI(
                temperature=0, google_api_key=api_key, model="gemini-pro")
        elif llm_provider == "DeepSeek (OpenAI compatible)":
            if not api_key:
                api_key = DEFAULT_DEEPSEEK_API_KEY if DEFAULT_DEEPSEEK_API_KEY else DEFAULT_OPENAI_API_KEY
            if not api_key:
                raise ValueError(_("api_key_error", provider="DeepSeek"))
            openai_api_base = os.getenv("OPENAI_API_BASE")
            if openai_api_base:
                llm = ChatOpenAI(temperature=0, openai_api_key=api_key,
                                 model_name="deepseek-coder", openai_api_base=openai_api_base)
            else:
                llm = ChatOpenAI(
                    temperature=0, openai_api_key=api_key, model_name="deepseek-coder")
        else:
            raise ValueError(_("unknown_provider", provider=llm_provider))

        llm_status_message_global = _("llm_success", provider=llm_provider)
        print(llm_status_message_global)
        # Limpa erros de LLM anteriores se a inicialização for bem-sucedida
        if llm_provider in setup_error_message_global:
            setup_error_message_global = setup_error_message_global.replace(
                f"Erro ao inicializar LLM com {llm_provider}:", "")
        return True
    except ImportError as e:
        error_msg = _("import_error", provider=llm_provider, error=str(e))
        print(error_msg)
        setup_error_message_global += error_msg + "\n"
        llm_status_message_global = error_msg
        llm = None
        return False
    except Exception as e:
        error_msg = _("init_error", provider=llm_provider, error=str(e))
        print(error_msg)
        setup_error_message_global += error_msg + "\n"
        llm_status_message_global = error_msg
        llm = None
        return False


def connect_db_and_setup_components(db_user, db_password, db_host, db_port, db_name, api_key_ui, llm_provider_ui, language=None):
    global db_engine, db, llm, agent_executor, chat_memory, current_language
    global current_db_config_display, connection_status_message_global, setup_error_message_global, llm_status_message_global

    # Atualiza o idioma se fornecido
    if language and language in TRANSLATIONS:
        current_language = language

    connection_status_message_local = ""
    current_db_config_display = _(
        "connecting_to", user=db_user, host=db_host, port=db_port, db_name=db_name)

    llm_initialized = _initialize_llm(api_key_ui, llm_provider_ui)
    database_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    try:
        db_engine = create_engine(database_url)
        with db_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db = SQLDatabase(engine=db_engine)
        connection_status_message_local = _(
            "connection_success", db_name=db_name, host=db_host)
        print(connection_status_message_local)

        if llm_initialized and db:
            try:
                # Inicializar a memória da conversa
                chat_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    input_key="input",
                    return_messages=True
                )

                # Determinar o agent_type com base no provedor LLM
                agent_type_kwargs = {}
                if llm_provider_ui in ["OpenAI", "DeepSeek (OpenAI compatible)"]:
                    agent_type_kwargs['agent_type'] = "openai-tools"

                # Adicionar informação sobre o idioma no prompt do sistema para o LLM
                system_message = f"You are an SQL agent that helps users query databases. Please respond in {current_language}."

                # Configurar opções avançadas do agente
                # Número padrão de linhas a retornar
                top_k = int(os.getenv("SQLORD_TOP_K", "50"))
                max_iterations = int(os.getenv("SQLORD_MAX_ITERATIONS", "15"))
                max_execution_time = float(os.getenv("SQLORD_MAX_EXECUTION_TIME", "60")) if os.getenv(
                    "SQLORD_MAX_EXECUTION_TIME") else None

                # Prefixo personalizado para o prompt (opcional)
                custom_prefix = os.getenv("SQLORD_CUSTOM_PREFIX", None)

                # Configurar callbacks para logging melhorado (opcional)
                agent_executor_kwargs = {
                    "handle_parsing_errors": True,
                    "return_intermediate_steps": True,
                }

                # Criar o SQL Agent com configurações avançadas
                agent_executor = create_sql_agent(
                    llm=llm,
                    db=db,
                    **agent_type_kwargs,
                    memory=chat_memory,
                    verbose=True,
                    system_message=system_message,
                    top_k=top_k,  # Número de linhas a retornar por padrão
                    max_iterations=max_iterations,  # Evitar loops infinitos
                    max_execution_time=max_execution_time,  # Timeout em segundos
                    prefix=custom_prefix,  # Prefix customizado se definido
                    agent_executor_kwargs=agent_executor_kwargs
                )

                print(
                    f"Agente SQL com memória inicializado com sucesso (top_k={top_k}, max_iterations={max_iterations}).")
                connection_status_message_local += _(
                    "agent_ready", kwargs=f"top_k={top_k}, iterations={max_iterations}")
            except Exception as e:
                error_msg = _("agent_init_error", error=str(e))
                print(error_msg)
                connection_status_message_local += f"\n{error_msg}"
                agent_executor = None
                chat_memory = None
        elif not llm_initialized:
            connection_status_message_local += _(
                "agent_fail", status=llm_status_message_global)
            agent_executor = None
            chat_memory = None

        current_db_config_display = _(
            "connected_to", user=db_user, host=db_host, port=db_port, db_name=db_name)

    except Exception as e:
        error_msg = _("db_error", error=str(e))
        print(error_msg)
        db = None
        agent_executor = None
        chat_memory = None
        current_db_config_display = _("connection_fail")
        connection_status_message_local = error_msg

    connection_status_message_global = connection_status_message_local
    return connection_status_message_local, current_db_config_display, llm_status_message_global, _get_chat_description()


def get_sql_and_result(user_question: str):
    """
    Processa uma pergunta do usuário em linguagem natural usando um agente SQL com memória,
    extrai a consulta SQL gerada e o resultado direto da execução da consulta.
    """
    global agent_executor, connection_status_message_global, setup_error_message_global, db

    if not agent_executor:
        error_detail = _("agent_not_initialized")
        if not db:
            error_detail += " " + _("db_not_connected_error")
        if not llm:
            error_detail += " " + _("llm_not_configured_error")
        if not chat_memory:
            error_detail += " " + _("memory_not_configured")
        if setup_error_message_global:
            error_detail += "\n" + \
                _("setup_errors", errors=setup_error_message_global)
        if llm_status_message_global and "Erro" in llm_status_message_global:
            error_detail += "\n" + \
                _("llm_status_error", status=llm_status_message_global)
        if connection_status_message_global and ("Falha na conexão" in connection_status_message_global or "Erro" in connection_status_message_global):
            error_detail += "\n" + \
                _("db_connection_status", status=connection_status_message_global)
        return f"Erro: {error_detail}", _("check_config")

    try:
        response = agent_executor.invoke(
            {"input": user_question, "chat_history": chat_memory.chat_memory.messages})
        print(f"DEBUG: Resposta completa do Agente SQL: {response}")

        generated_sql = _("no_sql_query")
        query_db_result = _("no_query_result")

        # Extrair a última query SQL e seu resultado dos intermediate_steps
        if response.get('intermediate_steps') and isinstance(response['intermediate_steps'], list):
            for step in reversed(response['intermediate_steps']):
                action, observation = step
                if hasattr(action, 'tool') and 'sql' in action.tool.lower():
                    if hasattr(action, 'tool_input') and action.tool_input:
                        if isinstance(action.tool_input, str):
                            generated_sql = action.tool_input
                        elif isinstance(action.tool_input, dict) and 'query' in action.tool_input:
                            generated_sql = action.tool_input['query']
                        else:
                            generated_sql = str(action.tool_input)

                        query_db_result = str(observation)
                        print(
                            f"DEBUG: SQL extraído da ação do agente: '{generated_sql}'")
                        print(
                            f"DEBUG: Resultado da query (observação do agente): '{query_db_result}'")
                        break

        if generated_sql == _("no_sql_query") and 'output' in response:
            query_db_result = response['output']
            generated_sql = _("agent_direct_response")

        if generated_sql != _("no_sql_query") and generated_sql != _("agent_direct_response"):
            if db and hasattr(db, 'dialect') and db.dialect == "postgresql":
                generated_sql = _apply_postgres_interval_fix(generated_sql)

        return generated_sql, query_db_result

    except Exception as e:
        error_msg = _("process_error", error=str(e))
        print(error_msg)
        return error_msg, ""


def _get_chat_description():
    global setup_error_message_global, connection_status_message_global, current_db_config_display, llm_status_message_global, agent_executor, chat_memory

    description = _("chat_intro") + "\n"
    description += _("db_config", config=current_db_config_display) + "\n"

    if llm:
        description += _("llm_status", status=llm_status_message_global) + "\n"
    else:
        description += _("llm_not_operational",
                         status=llm_status_message_global) + "\n"

    if setup_error_message_global:
        description += _("config_alerts",
                         alerts=setup_error_message_global) + "\n"

    if "Falha na conexão" in connection_status_message_global or "Aguardando configuração" in connection_status_message_global or "Erro" in connection_status_message_global:
        description += _("db_status",
                         status=connection_status_message_global) + "\n"
    elif db and agent_executor:
        description += _("db_connected") + "\n"
    elif db and not agent_executor and llm:
        description += _("db_connected_agent_fail") + "\n"
    elif not db:
        description += _("db_not_connected") + "\n"

    return description


# Adicionar às traduções
TRANSLATIONS["pt-BR"].update({
    "chat_header": "Conversa com o Banco de Dados",
    "chat_placeholder": "Digite sua pergunta sobre o banco de dados...",
    "chat_button": "Enviar",
    "chat_example": "Exemplo: Quais tabelas existem neste banco de dados?",
    "sql_prefix": "📊 SQL Gerado: \n```sql\n",
    "sql_suffix": "\n```\n",
    "viz_prefix": "📈 Visualização: ",
    "no_data": "Nenhum dado para exibição",
    "advanced_settings": "Configurações Avançadas",
    "top_k_label": "Linhas a retornar (top_k)",
    "max_iterations_label": "Máximo de iterações do agente",
    "max_execution_time_label": "Tempo máximo de execução (segundos)",
    "custom_prefix_label": "Prefixo personalizado para o prompt",
    "advanced_settings_info": "Configurações avançadas aplicadas: top_k={top_k}, max_iterations={max_iterations}"
})

TRANSLATIONS["en-US"].update({
    "chat_header": "Database Conversation",
    "chat_placeholder": "Type your question about the database...",
    "chat_button": "Send",
    "chat_example": "Example: What tables exist in this database?",
    "sql_prefix": "📊 Generated SQL: \n```sql\n",
    "sql_suffix": "\n```\n",
    "viz_prefix": "📈 Visualization: ",
    "no_data": "No data to display",
    "advanced_settings": "Advanced Settings",
    "top_k_label": "Rows to return (top_k)",
    "max_iterations_label": "Maximum agent iterations",
    "max_execution_time_label": "Maximum execution time (seconds)",
    "custom_prefix_label": "Custom prompt prefix",
    "advanced_settings_info": "Applied advanced settings: top_k={top_k}, max_iterations={max_iterations}"
})

# Histórico de chat como global para persistir entre interações
chat_history = []


# Configurações para visualização de dados
MAX_ROWS_FOR_VISUALIZATION = 1000  # Limite de linhas para visualização
MAX_COLS_FOR_VISUALIZATION = 150   # Limite de colunas para visualização


def determine_visualization_type(df):
    """
    Analisa um DataFrame e determina o tipo de visualização mais adequado.
    """
    # Se não houver dados suficientes para visualização
    if df.empty or len(df.columns) < 1:
        return {"type": "none", "message": _("insufficient_data")}

    # Se tiver muitas colunas, difícil visualizar bem
    if len(df.columns) > MAX_COLS_FOR_VISUALIZATION:
        return {"type": "table", "message": _("too_many_columns", count=len(df.columns))}

    # Classificar colunas por tipo
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(
        include=['datetime', 'datetimetz']).columns.tolist()

    # Tentar detectar colunas de data/hora em formato string
    for col in categorical_cols:
        if df[col].dtype == 'object':
            # Tenta converter para datetime para ver se é uma coluna de data disfarçada
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass

    # Determinar o tipo de visualização com base nos tipos de dados

    # Caso 1: Dados temporais (séries temporais)
    if datetime_cols and numeric_cols:
        return {
            "type": "line",
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "color": categorical_cols[0] if categorical_cols else None,
            "title": _("evolution_of", column=numeric_cols[0])
        }

    # Caso 2: Comparação entre categorias (barras)
    elif categorical_cols and numeric_cols:
        return {
            "type": "bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "color": categorical_cols[1] if len(categorical_cols) > 1 else None,
            "title": _("by", numeric=numeric_cols[0], category=categorical_cols[0])
        }

    # Caso 3: Distribuição de valores numéricos (histograma)
    elif numeric_cols and len(numeric_cols) >= 1:
        return {
            "type": "histogram",
            "x": numeric_cols[0],
            "color": categorical_cols[0] if categorical_cols else None,
            "title": _("distribution_of", column=numeric_cols[0])
        }

    # Caso 4: Correlação entre variáveis numéricas (scatter)
    elif len(numeric_cols) >= 2:
        return {
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "color": categorical_cols[0] if categorical_cols else None,
            "title": _("relation_between", column1=numeric_cols[0], column2=numeric_cols[1])
        }

    # Caso 5: Apenas categorias (pizza ou barras)
    elif categorical_cols and len(categorical_cols) >= 1:
        if len(df[categorical_cols[0]].unique()) <= 50:  # Alterado de 10 para 50
            return {
                "type": "pie",
                "names": categorical_cols[0],
                "title": _("distribution_of", column=categorical_cols[0])
            }
        else:
            return {
                "type": "bar",
                "x": categorical_cols[0],
                "title": _("count_of", column=categorical_cols[0])
            }

    # Caso padrão: tabela
    return {"type": "table", "message": _("data_not_suitable")}


def create_visualization(df, viz_config):
    """
    Cria uma visualização Plotly com base na configuração determinada.
    """
    try:
        if viz_config["type"] == "none" or viz_config["type"] == "table":
            return None

        if viz_config["type"] == "line":
            fig = px.line(
                df,
                x=viz_config["x"],
                y=viz_config["y"],
                color=viz_config.get("color"),
                title=viz_config.get("title", _("line_chart"))
            )

        elif viz_config["type"] == "bar":
            fig = px.bar(
                df,
                x=viz_config["x"],
                y=viz_config.get("y"),  # y pode ser None em alguns casos
                color=viz_config.get("color"),
                title=viz_config.get("title", _("bar_chart"))
            )

        elif viz_config["type"] == "histogram":
            fig = px.histogram(
                df,
                x=viz_config["x"],
                color=viz_config.get("color"),
                title=viz_config.get("title", _("histogram"))
            )

        elif viz_config["type"] == "scatter":
            fig = px.scatter(
                df,
                x=viz_config["x"],
                y=viz_config["y"],
                color=viz_config.get("color"),
                title=viz_config.get("title", _("scatter_plot"))
            )

        elif viz_config["type"] == "pie":
            # Agrupa e conta para criar o gráfico de pizza
            count_df = df[viz_config["names"]].value_counts().reset_index()
            count_df.columns = [viz_config["names"], 'count']

            fig = px.pie(
                count_df,
                names=viz_config["names"],
                values='count',
                title=viz_config.get("title", _("pie_chart"))
            )

        # Ajustes gerais da figura
        fig.update_layout(
            template="plotly_white",
            legend_title_text=_("legend")
        )

        return fig

    except Exception as e:
        print(_("viz_error", error=str(e)))
        return None


def convert_sql_result_to_dataframe(sql_result):
    """
    Converte o resultado de uma consulta SQL em um DataFrame pandas.
    """
    try:
        # Tenta interpretar como JSON (formato comum para resultados)
        try:
            data = json.loads(sql_result)
            if isinstance(data, list) and len(data) > 0:
                return pd.DataFrame(data)
        except:
            pass

        # Tenta interpretar como CSV/TSV
        try:
            return pd.read_csv(StringIO(sql_result), sep=None, engine='python')
        except:
            pass

        # Tenta interpretar como texto tabulado ou espaçado
        lines = sql_result.strip().split('\n')
        if len(lines) >= 2:  # Precisa de pelo menos header + uma linha
            # Supõe que a primeira linha é o cabeçalho
            header = lines[0].split()
            data = []
            for line in lines[1:]:
                values = line.split()
                if len(values) == len(header):
                    data.append(values)

            if data:
                return pd.DataFrame(data, columns=header)

        return None
    except Exception as e:
        print(_("convert_error", error=str(e)))
        return None


def get_sql_and_result_with_viz(user_question: str):
    """
    Processa uma pergunta e gera SQL, resultado e visualização.

    Args:
        user_question: A pergunta do usuário em linguagem natural

    Returns:
        Tupla com (sql_gerado, resultado_query, figura_plotly, descrição_visualização)
    """
    # Obter SQL e resultado usando a função existente
    generated_sql, query_db_result = get_sql_and_result(user_question)

    # Se temos um resultado de query bem-sucedido, tentamos visualizar
    if query_db_result and not query_db_result.startswith("Erro:") and not query_db_result == _("check_config"):
        # Converter o resultado para DataFrame
        df = convert_sql_result_to_dataframe(query_db_result)

        if df is not None and not df.empty and len(df) <= MAX_ROWS_FOR_VISUALIZATION:
            # Determinar o tipo de visualização apropriado
            viz_config = determine_visualization_type(df)
            # Criar a visualização
            fig = create_visualization(df, viz_config)

            if fig:
                return generated_sql, query_db_result, fig, viz_config.get("title", _("viz_title"))
            else:
                message = viz_config.get("message", _("cannot_convert"))
                return generated_sql, query_db_result, None, message
        elif df is not None and len(df) > MAX_ROWS_FOR_VISUALIZATION:
            return generated_sql, query_db_result, None, _("too_large_dataset", rows=len(df), limit=MAX_ROWS_FOR_VISUALIZATION)
        else:
            return generated_sql, query_db_result, None, _("cannot_convert")

    # Se não conseguimos visualizar, retornamos sem figura
    return generated_sql, query_db_result, None, _("no_viz_data")


def process_chat_message(message, history):
    """
    Processa a mensagem do chat, gera a resposta com SQL e resultado integrados.
    Retorna a resposta para exibição no chatbot no formato correto para type='messages'.
    """
    global chat_history

    if not message.strip():
        return "", history

    # Obter SQL, resultado e visualização
    sql, result, fig, viz_desc = get_sql_and_result_with_viz(message)

    # Construir resposta formatada
    response = ""

    # Adicionar SQL gerado, se disponível
    if sql and sql != _("no_sql_query") and sql != _("agent_direct_response"):
        response += _("sql_prefix") + sql + _("sql_suffix")

    # Adicionar resultado
    if result:
        response += result

    # Informações de visualização, se disponível
    if fig:
        # Converter a figura para HTML para exibir no chat
        fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        response += f"\n\n{_('viz_prefix')} {viz_desc}\n\n"
        response += f"<div>{fig_html}</div>"
    elif viz_desc and viz_desc != _("no_viz_data"):
        response += f"\n\n{_('viz_prefix')} {viz_desc}"

    # Criar mensagens no formato correto para type='messages' (dicionários com 'role' e 'content')
    user_message = {"role": "user", "content": message}
    assistant_message = {"role": "assistant", "content": response}

    # Atualizar histórico com o formato correto
    updated_history = history + [user_message, assistant_message]
    chat_history = updated_history

    return "", updated_history


def main():
    print("Iniciando SQLord...")
    global connection_status_message_global, current_db_config_display, llm_status_message_global, current_language, chat_history

    with gr.Blocks() as iface:
        # Título da aplicação
        app_title = gr.Markdown(_("app_title"))

        # Seletor de idioma no topo
        with gr.Row():
            language_selector = gr.Dropdown(
                choices=list(TRANSLATIONS.keys()),
                value=current_language,
                label="Idioma / Language",
                interactive=True
            )

        # Criar as abas - não tentaremos alterar seus nomes após a criação
        with gr.Tabs() as tabs:
            # Interface de chat simplificada
            with gr.TabItem(_("chat_tab")) as chat_tab:
                status_info = gr.Markdown(value=_get_chat_description)

                # Interface de chatbot com type='messages'
                chatbot = gr.Chatbot(
                    value=chat_history,
                    show_label=False,
                    height=400,
                    type='messages'
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder=_("chat_placeholder"),
                        show_label=False,
                        scale=9
                    )
                    submit = gr.Button(_("chat_button"), scale=1)

                gr.Examples(
                    examples=[
                        _("chat_example"),
                        "Quais são as 5 cidades mais populosas?",
                        "Mostre os usuários criados nos últimos 30 dias",
                    ],
                    inputs=msg
                )

                # Botão para limpar conversa - atualizado para retornar lista vazia
                clear = gr.Button("Limpar Conversa")
                clear.click(lambda: [], outputs=[chatbot], show_progress=False)

                # Processar mensagem
                submit.click(
                    process_chat_message,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                )
                msg.submit(
                    process_chat_message,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                )

            # Aba de configuração
            with gr.TabItem(_("config_tab")) as config_tab:
                db_config_title = gr.Markdown(_("db_config_title"))

                db_user_input = gr.Textbox(
                    label=_("db_user_label"), value=DEFAULT_DB_USER)
                db_password_input = gr.Textbox(
                    label=_("db_password_label"), type="password", value=DEFAULT_DB_PASSWORD)
                db_host_input = gr.Textbox(
                    label=_("db_host_label"), value=DEFAULT_DB_HOST)
                db_port_input = gr.Textbox(
                    label=_("db_port_label"), value=DEFAULT_DB_PORT)
                db_name_input = gr.Textbox(
                    label=_("db_name_label"), value=DEFAULT_DB_NAME)

                llm_config_title = gr.Markdown(_("llm_config_title"))
                llm_provider_input = gr.Dropdown(
                    choices=LLM_PROVIDERS, label=_("llm_provider_label"), value=LLM_PROVIDERS[0])
                api_key_input = gr.Textbox(
                    label=_("api_key_label"),
                    type="password",
                    placeholder=_("api_key_placeholder")
                )

                connect_button = gr.Button(_("connect_button"))

                status_title = gr.Markdown(_("status_title"))
                llm_status_output = gr.Textbox(
                    label=_("llm_status_label"), interactive=False, lines=2, value=llm_status_message_global)
                connection_status_output = gr.Textbox(
                    label=_("connection_status_label"), interactive=False, lines=2, value=connection_status_message_global)
                current_config_output = gr.Textbox(
                    label=_("current_config_label"), interactive=False, value=current_db_config_display)

                # Adicionar seção de configurações avançadas
                advanced_settings_title = gr.Markdown(
                    "### " + _("advanced_settings"))

                with gr.Accordion("Configurações avançadas", open=False):
                    top_k_input = gr.Slider(
                        minimum=5, maximum=500, value=50, step=5,
                        label=_("top_k_label")
                    )
                    max_iterations_input = gr.Slider(
                        minimum=5, maximum=50, value=15, step=1,
                        label=_("max_iterations_label")
                    )
                    max_execution_time_input = gr.Slider(
                        minimum=10, maximum=300, value=60, step=5,
                        label=_("max_execution_time_label")
                    )
                    custom_prefix_input = gr.Textbox(
                        label=_("custom_prefix_label"),
                        placeholder="Deixe em branco para usar o padrão",
                        lines=3
                    )

                # Adicionar botão para visualizar esquema do banco
                db_schema_button = gr.Button("Ver Esquema do Banco")
                db_schema_output = gr.TextArea(
                    label="Esquema do Banco de Dados",
                    interactive=False,
                    visible=True,
                    lines=10
                )

                # Função para atualizar o esquema
                db_schema_button.click(
                    fn=get_db_table_info,
                    outputs=db_schema_output
                )

                # Função para conectar usando o idioma atual
                def connect_with_language(db_user, db_password, db_host, db_port, db_name, api_key_ui, llm_provider_ui):
                    return connect_db_and_setup_components(
                        db_user, db_password, db_host, db_port, db_name,
                        api_key_ui, llm_provider_ui, current_language
                    )

                connect_button.click(
                    fn=connect_with_language,
                    inputs=[db_user_input, db_password_input, db_host_input,
                            db_port_input, db_name_input, api_key_input, llm_provider_input],
                    outputs=[connection_status_output, current_config_output,
                             llm_status_output, status_info]
                )

        # SOLUÇÃO DEFINITIVA: Usar update_ui_on_language_change para atualizar textos
        def update_ui_on_language_change(language):
            """
            Atualiza a interface com base no novo idioma selecionado.
            Usa gr.update() para garantir que as propriedades dos componentes sejam atualizadas corretamente.
            """
            # Altera o idioma global
            set_language(language)

            # Retorna os novos valores para os componentes
            return {
                app_title: _("app_title"),
                status_info: _get_chat_description(),
                submit: _("chat_button"),
                db_config_title: _("db_config_title"),
                llm_config_title: _("llm_config_title"),
                status_title: _("status_title"),
                # Atualizar propriedades de componentes
                msg: gr.update(placeholder=_("chat_placeholder")),
                db_user_input: gr.update(label=_("db_user_label")),
                db_password_input: gr.update(label=_("db_password_label")),
                db_host_input: gr.update(label=_("db_host_label")),
                db_port_input: gr.update(label=_("db_port_label")),
                db_name_input: gr.update(label=_("db_name_label")),
                llm_provider_input: gr.update(label=_("llm_provider_label")),
                api_key_input: gr.update(label=_("api_key_label"), placeholder=_("api_key_placeholder")),
                connect_button: _("connect_button"),
                llm_status_output: gr.update(label=_("llm_status_label")),
                connection_status_output: gr.update(label=_("connection_status_label")),
                current_config_output: gr.update(label=_("current_config_label")),
                clear: "Limpar Conversa" if language == "pt-BR" else "Clear Chat",
                advanced_settings_title: "### " + _("advanced_settings"),
                top_k_input: gr.update(label=_("top_k_label")),
                max_iterations_input: gr.update(label=_("max_iterations_label")),
                max_execution_time_input: gr.update(label=_("max_execution_time_label")),
                custom_prefix_input: gr.update(label=_("custom_prefix_label")),
                db_schema_button: "Ver Esquema do Banco" if language == "pt-BR" else "View Database Schema",
            }

        # Registrar a função para executar quando o idioma mudar
        language_selector.change(
            fn=update_ui_on_language_change,
            inputs=language_selector,
            outputs=[
                app_title, status_info, submit, db_config_title, llm_config_title,
                status_title, msg, db_user_input, db_password_input, db_host_input,
                db_port_input, db_name_input, llm_provider_input, api_key_input,
                connect_button, llm_status_output, connection_status_output,
                current_config_output, clear, advanced_settings_title, top_k_input,
                max_iterations_input, max_execution_time_input, custom_prefix_input,
                db_schema_button
            ]
        )

    print("Lançando interface Gradio...")
    iface.launch(mcp_server=True)


# Função para obter informações sobre as tabelas do banco de dados
def get_db_table_info():
    """Obtém informações detalhadas sobre as tabelas do banco de dados conectado."""
    global db

    if not db:
        return _("db_not_connected_error")

    try:
        # Usar a função interna do SQLDatabase para obter informações das tabelas
        table_info = db.get_table_info()
        return table_info
    except Exception as e:
        error_msg = f"Erro ao obter informações das tabelas: {str(e)}"
        print(error_msg)
        return error_msg


if __name__ == "__main__":
    main()
