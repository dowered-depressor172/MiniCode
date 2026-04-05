import readline from 'node:readline'
import process from 'node:process'
import { AnthropicModelAdapter } from './anthropic-adapter.js'
import {
  completeSlashCommand,
  findMatchingSlashCommands,
  tryHandleLocalCommand,
} from './cli-commands.js'
import { loadRuntimeConfig } from './config.js'
import { maybeHandleManagementCommand } from './manage-cli.js'
import { MockModelAdapter } from './mock-model.js'
import { PermissionManager } from './permissions.js'
import { buildSystemPrompt } from './prompt.js'
import { createDefaultToolRegistry, hydrateMcpTools } from './tools/index.js'
import type { ChatMessage } from './types.js'
import { renderBanner } from './ui.js'
import { runTtyApp } from './tty-app.js'
import { runAgentTurn } from './agent-loop.js'

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  if (await maybeHandleManagementCommand(process.cwd(), argv)) {
    return
  }

  const isInteractiveTerminal = Boolean(process.stdin.isTTY && process.stdout.isTTY)
  let runtime = null
  try {
    runtime = await loadRuntimeConfig()
  } catch {
    runtime = null
  }

  const tools = await createDefaultToolRegistry({
    cwd: process.cwd(),
    runtime,
  })
  const mcpHydration = hydrateMcpTools({
    cwd: process.cwd(),
    runtime,
    tools,
  }).catch(() => {
    // Keep startup resilient even if some MCP servers fail.
  })
  const permissions = new PermissionManager(process.cwd())
  await permissions.whenReady()
  const model =
    process.env.MINI_CODE_MODEL_MODE === 'mock'
      ? new MockModelAdapter()
      : new AnthropicModelAdapter(tools, loadRuntimeConfig)
  let messages: ChatMessage[] = [
    {
      role: 'system',
      content: await buildSystemPrompt(process.cwd(), permissions.getSummary(), {
        skills: tools.getSkills(),
        mcpServers: tools.getMcpServers(),
      }),
    },
  ]

  try {
    if (isInteractiveTerminal) {
      await runTtyApp({
        runtime,
        tools,
        model,
        messages,
        cwd: process.cwd(),
        permissions,
      })
      return
    }

    console.log(
      renderBanner(runtime, process.cwd(), permissions.getSummary(), {
        transcriptCount: 0,
        messageCount: messages.length,
        skillCount: tools.getSkills().length,
        mcpTotalCount: tools.getMcpServers().length,
        mcpConnectedCount: tools
          .getMcpServers()
          .filter(server => server.status === 'connected').length,
        mcpConnectingCount: tools
          .getMcpServers()
          .filter(server => server.status === 'connecting').length,
        mcpErrorCount: tools
          .getMcpServers()
          .filter(server => server.status === 'error').length,
      }),
    )
    console.log('')

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      completer: completeSlashCommand,
    })

    if (isInteractiveTerminal) {
      rl.setPrompt('minicode> ')
      rl.prompt()
    }

    for await (const rawInput of rl) {
      const input = rawInput.trim()
      if (!input) {
        if (isInteractiveTerminal) rl.prompt()
        continue
      }
      if (input === '/exit') break

      try {
        if (input === '/tools') {
          console.log(
            `\n${tools.list().map(tool => `${tool.name}: ${tool.description}`).join('\n')}\n`,
          )
          if (isInteractiveTerminal) rl.prompt()
          continue
        }

        const localCommandResult = await tryHandleLocalCommand(input, { tools })
        if (localCommandResult !== null) {
          console.log(`\n${localCommandResult}\n`)
          if (isInteractiveTerminal) rl.prompt()
          continue
        }

        if (input.startsWith('/')) {
          const matches = findMatchingSlashCommands(input)
          if (matches.length > 0) {
            console.log(`\n未识别命令。你是不是想输入：\n${matches.join('\n')}\n`)
          } else {
            console.log(`\n未识别命令。输入 /help 查看可用命令。\n`)
          }
          if (isInteractiveTerminal) rl.prompt()
          continue
        }
      } catch (error) {
        console.log(
          `\n${error instanceof Error ? error.message : String(error)}\n`,
        )
        if (isInteractiveTerminal) rl.prompt()
        continue
      }

      messages[0] = {
        role: 'system',
        content: await buildSystemPrompt(process.cwd(), permissions.getSummary(), {
          skills: tools.getSkills(),
          mcpServers: tools.getMcpServers(),
        }),
      }
      messages = [...messages, { role: 'user', content: input }]
      permissions.beginTurn()
      try {
        messages = await runAgentTurn({
          model,
          tools,
          messages,
          cwd: process.cwd(),
          permissions,
        })
      } catch (error) {
        const message =
          error instanceof Error ? error.message : String(error)
        messages = [
          ...messages,
          {
            role: 'assistant',
            content: `请求失败: ${message}`,
          },
        ]
      } finally {
        permissions.endTurn()
      }

      const lastAssistant = [...messages]
        .reverse()
        .find(message => message.role === 'assistant')

      if (lastAssistant?.role === 'assistant') {
        console.log(`\n${lastAssistant.content}\n`)
      }

      if (isInteractiveTerminal) rl.prompt()
    }

    try {
      rl.close()
    } catch {
      // Ignore double-close during EOF teardown.
    }
  } finally {
    await mcpHydration
    await tools.dispose()
  }
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
