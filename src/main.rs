use std::{
    fs::{File, OpenOptions},
    io::{BufRead, Read},
};

use futures_util::StreamExt;
use rig::{
    OneOrMany,
    agent::Text,
    client::CompletionClient,
    completion::{CompletionModel, CompletionRequest, ToolDefinition},
    message::{Message, ToolResult, ToolResultContent, UserContent},
    providers::gemini::{
        Client,
        completion::gemini_api_types::{AdditionalParameters, GenerationConfig},
    },
    streaming::StreamedAssistantContent,
};

struct HistoryManager<W: std::io::Write> {
    writer: W,
    history: Vec<Message>,
}

impl<W: std::io::Write> HistoryManager<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            history: vec![],
        }
    }

    fn new_with_history(writer: W, history: Vec<Message>) -> Self {
        Self { writer, history }
    }

    fn add_user_message(&mut self, text: String) {
        let message = Message::User {
            content: OneOrMany::one(UserContent::Text(Text { text })),
        };

        self.writer
            .write_all(format!("{}\n", serde_json::to_string(&message).unwrap()).as_bytes())
            .unwrap();
        self.writer.flush().unwrap();

        self.history.push(message);
    }

    fn handle_tool_call_result(&mut self, result: ToolResult) {
        let message = Message::User {
            content: OneOrMany::one(rig::message::UserContent::ToolResult(result)),
        };

        self.writer
            .write_all(format!("{}\n", serde_json::to_string(&message).unwrap()).as_bytes())
            .unwrap();
        self.writer.flush().unwrap();

        self.history.push(message);
    }

    fn handle_completion<R>(&mut self, completion: StreamedAssistantContent<R>) {
        let message = match completion {
            StreamedAssistantContent::Text(text) => Message::Assistant {
                id: None,
                content: OneOrMany::one(rig::message::AssistantContent::Text(text)),
            },
            StreamedAssistantContent::ToolCall(call) => Message::Assistant {
                id: None,
                content: OneOrMany::one(rig::message::AssistantContent::ToolCall(call)),
            },
            StreamedAssistantContent::Final(_) => {
                return;
            }
            _ => todo!(),
        };

        self.writer
            .write_all(format!("{}\n", serde_json::to_string(&message).unwrap()).as_bytes())
            .unwrap();
        self.writer.flush().unwrap();

        self.history.push(message);
    }

    fn get_history(&self) -> OneOrMany<Message> {
        OneOrMany::many(self.history.clone()).unwrap()
    }
}

#[tokio::main]
async fn main() {
    let client = Client::from_env();
    let gemini = client.completion_model("gemini-2.5-flash-lite");

    let mut current_history = if let Ok(file) = File::open("conversation_log.jsonl") {
        let reader = std::io::BufReader::new(file);
        let history: Vec<Message> = reader
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
            .collect();

        HistoryManager::new_with_history(
            OpenOptions::new()
                .write(true)
                .append(true)
                .open("conversation_log.jsonl")
                .unwrap(),
            history,
        )
    } else {
        let file = File::create("conversation_log.jsonl").unwrap();
        HistoryManager::new(file)
    };

    let input = std::io::stdin();
    let mut lines = input.lines();

    let tools = vec![
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a given location.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for."
                    }
                },
                "required": ["location"]
            }),
        }
    ];

    let mut resume = if let Some(Message::Assistant { content, .. }) = current_history.history.last()
    && let rig::message::AssistantContent::ToolCall(tool_call) = content.first()
    {
        println!(
            "Resuming from last tool call: {} with arguments {}",
            tool_call.function.name, tool_call.function.arguments
        );

        let mut result = String::new();
        File::open("tool_call.json").unwrap().read_to_string(&mut result).unwrap();
        println!("Tool call result loaded from tool_call.json: {}", result);

        current_history.handle_tool_call_result(ToolResult {
            id: tool_call.id.clone(),
            call_id: tool_call.call_id.clone(),
            content: OneOrMany::one(ToolResultContent::Text(Text { text: result })),
        });

        true
    } else {
        false
    };

    'outer: loop {
        if resume {
            resume = false;
        } else {
            print!("\n> ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            let line = lines.next().unwrap().unwrap();
            if line.trim().is_empty() {
                break;
            }

            current_history.add_user_message(line);
        }

        let gen_cfg = GenerationConfig::default();
        let cfg = AdditionalParameters::default().with_config(gen_cfg);
        let mut completion_result = gemini
            .stream(CompletionRequest {
                preamble: None,
                chat_history: current_history.get_history(),
                documents: vec![],
                tools: tools.clone(),
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: Some(serde_json::to_value(cfg).unwrap()),
            })
            .await
            .unwrap();

        while let Some(Ok(chunk)) = completion_result.next().await {
            let was_call = match &chunk {
                StreamedAssistantContent::Text(text) => {
                    print!("{}", text.text);
                    false
                }
                StreamedAssistantContent::ToolCall(call) => {
                    print!(
                        "\n[Tool Call: {} with arguments {}]\n",
                        call.function.name,
                        call.function.arguments
                    );

                    let mut tool_call_file = File::create("tool_call.json").unwrap();
                    serde_json::to_writer_pretty(&mut tool_call_file, &call).unwrap();

                    true
                }
                StreamedAssistantContent::Final(_) => {
                    false
                }
                o => todo!("Unhandled chunk type: {:?}", o),
            };

            current_history.handle_completion(chunk);

            if was_call {
                println!("\n--- Conversation ended due to tool call ---");
                break 'outer;
            }
        }
    }
}
