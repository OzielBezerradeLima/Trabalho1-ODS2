import { useMemo, useState, useRef } from 'react';
import { FileText, Plus, X, Upload, Sparkles, Target, Beaker, BarChart3, AlertTriangle, Link, Send, Loader2, User, Bot } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  isLoading?: boolean;
  context?: string;
}

interface UploadedFile {
  name: string;
  size: number;
  indexed: boolean;
}

interface UploadResponse {
  session_id: string;
  doc_name: string;
  chunks_count: number;
  collection_name: string;
}

const suggestedQuestions = [
  { icon: FileText, text: 'Faça um resumo completo deste artigo' },
  { icon: Target, text: 'Qual é o objetivo principal do estudo?' },
  { icon: Beaker, text: 'Quais métodos foram utilizados?' },
  { icon: BarChart3, text: 'Quais foram os principais resultados?' },
  { icon: AlertTriangle, text: 'Quais são as limitações do trabalho?' },
  { icon: Link, text: 'Quais referências são mais citadas?' },
];

export default function App() {
  const [file, setFile] = useState<UploadedFile | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [chunksCount, setChunksCount] = useState<number | null>(null);
  const [collectionName, setCollectionName] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [apiError, setApiError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000';

  const indexed = Boolean(file?.indexed && sessionId);

  const healthBadge = useMemo(() => {
    return indexed ? 'API conectada e documento indexado' : 'Aguardando indexacao';
  }, [indexed]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setApiError('');
      setFile({
        name: uploadedFile.name,
        size: uploadedFile.size,
        indexed: false
      });
      setSessionId('');
      setCollectionName('');
      setChunksCount(null);
      setMessages([]);
    }
  };

  const handleIndexDocument = async () => {
    if (!file || file.indexed || !fileInputRef.current?.files?.[0]) return;

    setIsIndexing(true);
    setApiError('');

    try {
      const payload = new FormData();
      payload.append('file', fileInputRef.current.files[0]);

      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: 'POST',
        body: payload,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ detail: 'Erro desconhecido ao indexar' }));
        throw new Error(data.detail || 'Falha ao indexar PDF');
      }

      const data: UploadResponse = await response.json();
      setSessionId(data.session_id);
      setChunksCount(data.chunks_count);
      setCollectionName(data.collection_name);
      setFile({ ...file, indexed: true });
      setMessages([]);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Falha ao indexar documento');
    } finally {
      setIsIndexing(false);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setSessionId('');
    setCollectionName('');
    setChunksCount(null);
    setMessages([]);
    setApiError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (presetQuestion?: string) => {
    const text = (presetQuestion ?? inputValue).trim();
    if (!text || !sessionId) return;

    setApiError('');

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: text
    };

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Buscando trechos relevantes e gerando resposta...',
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInputValue('');

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          question: text,
          top_k: 5,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ detail: 'Erro desconhecido no chat' }));
        throw new Error(data.detail || 'Falha ao consultar o chat');
      }

      const data = await response.json();

      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: data.answer, context: data.context, isLoading: false }
            : msg
        )
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Falha ao gerar resposta';
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: `Erro: ${message}`, isLoading: false }
            : msg
        )
      );
      setApiError(message);
    }
  };

  const handleQuestionClick = (question: string) => {
    void handleSendMessage(question);
  };

  const formatFileSize = (bytes: number) => {
    return `${(bytes / 1024).toFixed(1)} KB`;
  };

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex">
      {/* Sidebar */}
      <div className="w-[400px] bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-2xl font-bold">
            <span className="text-white">Article</span>
            <span className="text-teal-500">AI</span>
          </h1>
        </div>

        {/* Document Section */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-4 text-gray-400 text-sm uppercase tracking-wide">
              <FileText size={16} />
              <span>Documento</span>
            </div>

            {/* File Upload */}
            <div className="mb-4">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />

              {file ? (
                <div className="bg-gray-800 rounded-lg p-4 flex items-start gap-3">
                  <FileText className="text-gray-400 flex-shrink-0 mt-1" size={20} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm text-gray-200 truncate">{file.name}</p>
                      <button
                        onClick={handleRemoveFile}
                        className="text-gray-400 hover:text-gray-200 flex-shrink-0"
                      >
                        <X size={16} />
                      </button>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{formatFileSize(file.size)}</p>
                  </div>
                  <label
                    htmlFor="file-upload"
                    className="text-gray-400 hover:text-gray-200 cursor-pointer flex-shrink-0"
                  >
                    <Plus size={20} />
                  </label>
                </div>
              ) : (
                <label
                  htmlFor="file-upload"
                  className="border-2 border-dashed border-gray-700 rounded-lg p-8 flex flex-col items-center gap-2 cursor-pointer hover:border-teal-500 transition-colors"
                >
                  <Upload className="text-gray-400" size={32} />
                  <span className="text-sm text-gray-400">Clique para fazer upload</span>
                </label>
              )}
            </div>

            {/* Index Button */}
            {file && (
              <>
                <button
                  onClick={handleIndexDocument}
                  disabled={file.indexed || isIndexing}
                  className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-teal-800 disabled:opacity-50 text-white rounded-lg py-3 px-4 flex items-center justify-center gap-2 font-medium transition-colors"
                >
                  {isIndexing ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
                  {file.indexed ? 'Documento Indexado' : isIndexing ? 'Indexando...' : 'Indexar documento'}
                </button>

                {/* Index Details */}
                {file.indexed && (
                  <div className="mt-4 space-y-2 text-sm">
                    <div className="flex justify-between text-gray-400">
                      <span>arquivo</span>
                      <span className="text-gray-300 truncate ml-2">{file.name}</span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>chunks</span>
                      <span className="text-teal-500 flex items-center gap-1">
                        <Plus size={14} />
                        {chunksCount ?? 0} blocos
                      </span>
                    </div>
                    <div className="flex justify-between text-gray-400">
                      <span>collection</span>
                      <span className="text-gray-300 truncate ml-2">{collectionName || '-'}</span>
                    </div>
                  </div>
                )}
              </>
            )}
            {apiError && <p className="mt-3 text-sm text-rose-400">{apiError}</p>}
          </div>

          {/* Suggested Questions */}
          <div>
            <div className="flex items-center gap-2 mb-4 text-yellow-500 text-sm uppercase tracking-wide">
              <Sparkles size={16} />
              <span>Perguntas Sugeridas</span>
            </div>

            <div className="space-y-2">
              {suggestedQuestions.map((question, index) => {
                const Icon = question.icon;
                return (
                  <button
                    key={index}
                    onClick={() => handleQuestionClick(question.text)}
                    disabled={!indexed}
                    className="w-full bg-teal-700 hover:bg-teal-600 text-white rounded-lg py-3 px-4 flex items-center gap-3 text-left transition-colors"
                  >
                    <Icon size={18} />
                    <span className="text-sm">{question.text}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        {indexed && (
          <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-400">Artigo indexado:</span>
                <span className="text-teal-500">{file.name}</span>
                <span className="text-gray-600">• Faça sua primeira pergunta +</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span className="text-xs text-gray-400">{healthBadge}</span>
              </div>
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
          {messages.map((message) => (
            <div key={message.id} className="flex gap-4">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.type === 'user' ? 'bg-teal-600' : 'bg-purple-600'
              }`}>
                {message.type === 'user' ? <User size={18} /> : <Bot size={18} />}
              </div>
              <div className="flex-1">
                <div className={`inline-block px-4 py-3 rounded-lg ${
                  message.type === 'user'
                    ? 'bg-gray-800 text-gray-100'
                    : 'bg-gray-900 text-gray-300'
                }`}>
                  {message.isLoading ? (
                    <div className="flex items-center gap-2 text-teal-500">
                      <Loader2 className="animate-spin" size={16} />
                      <span>{message.content}</span>
                    </div>
                  ) : <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>}
                </div>
                {message.context && (
                  <details className="mt-2 text-xs text-gray-400 bg-gray-900 border border-gray-800 rounded-md p-2">
                    <summary className="cursor-pointer">Ver contexto recuperado</summary>
                    <p className="mt-2 whitespace-pre-wrap">{message.context}</p>
                  </details>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-800 p-6">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && void handleSendMessage()}
                placeholder="Pergunte sobre o artigo, peça um resumo, metodologia..."
                className="w-full bg-gray-900 border border-gray-700 rounded-lg py-4 px-6 pr-12 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-teal-500"
                disabled={!indexed}
              />
              <button
                onClick={() => void handleSendMessage()}
                disabled={!inputValue.trim() || !indexed}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-teal-500 hover:text-teal-400 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}