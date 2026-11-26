package ma.emsi.chorouk;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import java.util.List;

import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {

        System.out.println("=== TP4 - Test 5 : RAG avec récupération sur le Web ===");

        // 1. Vérification de la clé API Tavily
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");

        if (tavilyApiKey == null || tavilyApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable TAVILY_API_KEY n'est pas configurée."
            );
        }

        System.out.println("Clé Tavily détectée avec succès !");

        // 2. Création du moteur de recherche Tavily
        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyApiKey)
                        .build();

        System.out.println("TavilyWebSearchEngine créé avec succès !");

        // 3. Création du récupérateur de contenu Web
        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        System.out.println("WebSearchContentRetriever créé avec succès !");

        // PHASE 1 : Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("\n=== RAG à partir du PDF ===");
        System.out.println("Document rag.pdf chargé avec succès !");

        // Découpage du document en segments
        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // Modèle d'embeddings
        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        // Création des embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        // Magasin vectoriel
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés : "
                + embeddings.size());

        // PHASE 2 : Création du récupérateur PDF
        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever PDF créé avec succès !");

        // PHASE 3 : Routage vers le PDF et le Web

        // Les deux récupérateurs seront utilisés
        QueryRouter queryRouter =
                new DefaultQueryRouter(
                        pdfRetriever,
                        webRetriever
                );

        System.out.println("\n=== Routage PDF + Web ===");
        System.out.println("DefaultQueryRouter créé avec succès !");

        // Création du RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé avec succès !");

        // PHASE 4 : Configuration de Gemini
        String geminiApiKey = System.getenv("GEMINI_API_KEY");

        if (geminiApiKey == null || geminiApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable GEMINI_API_KEY n'est pas configurée."
            );
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        System.out.println("Modèle Gemini initialisé !");

        // Mémoire de conversation : 10 messages
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // Assistant connecté aux deux sources du RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG PDF + Web créé avec succès !");

        // PHASE 5 : Conversation interactive
        Scanner scanner = new Scanner(System.in);

        System.out.println("\n=== Conversation RAG PDF + Web ===");
        System.out.println("Tapez 'fin' pour terminer.");

        while (true) {

            System.out.print("\nVotre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("fin")) {
                System.out.println("Fin de la conversation.");
                break;
            }

            if (question.isBlank()) {
                System.out.println("Veuillez saisir une question.");
                continue;
            }

            String reponse = assistant.chat(question);

            System.out.println("\nRéponse de Gemini :");
            System.out.println(reponse);
        }



    }
}
