package ma.emsi.chorouk;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

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

import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Scanner;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.model.input.PromptTemplate;
import java.util.Collection;
import java.util.Map;
import java.util.Locale;

public class Test4 {

    public static void main(String[] args) {

        System.out.println(
                "=== TP4 - Test 5 : RAG avec récupération sur le Web ==="
        );

        // ==========================================
        // PHASE 1 : MOTEUR DE RECHERCHE WEB TAVILY
        // ==========================================

        // Vérification de la clé API Tavily
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");

        if (tavilyApiKey == null || tavilyApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable TAVILY_API_KEY n'est pas configurée."
            );
        }

        System.out.println("Clé Tavily détectée avec succès !");

        // Création du moteur de recherche Tavily
        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(tavilyApiKey)
                        .build();

        System.out.println(
                "TavilyWebSearchEngine créé avec succès !"
        );

        // Création du récupérateur Web
        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        System.out.println(
                "WebSearchContentRetriever créé avec succès !"
        );

        // ==========================================
        // PHASE 2 : CHARGEMENT DU DOCUMENT PDF
        // ==========================================

        System.out.println("\n=== RAG à partir du PDF ===");

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        "rag.pdf",
                        new ApacheTikaDocumentParser()
                );

        System.out.println(
                "Document rag.pdf chargé avec succès !"
        );

        // Découpage du document en segments
        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println(
                "Nombre de segments : " + segments.size()
        );

        // Modèle d'embeddings
        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        // Création des embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        // Magasin vectoriel en mémoire
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println(
                "Embeddings enregistrés : " + embeddings.size()
        );

        // ==========================================
        // PHASE 3 : CRÉATION DU RÉCUPÉRATEUR PDF
        // ==========================================

        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println(
                "ContentRetriever PDF créé avec succès !"
        );

        // ==========================================
        // PHASE 4 : CONFIGURATION DE GEMINI
        // ==========================================

        // Vérification de la clé API Gemini
        String geminiApiKey = System.getenv("GEMINI_API_KEY");

        if (geminiApiKey == null || geminiApiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable GEMINI_API_KEY n'est pas configurée."
            );
        }

        // Initialisation du modèle Gemini
        ChatModel model =
                GoogleGenAiChatModel.builder()
                        .apiKey(geminiApiKey)
                        .modelName("gemini-3.5-flash-lite")
                        .temperature(0.3)
                        .logRequests(true)
                        .logResponses(true)
                        .build();

        System.out.println("Modèle Gemini initialisé !");

        // ==========================================
        // PHASE 5 : ROUTAGE PDF + WEB
        // ==========================================

        System.out.println("\n=== Routage PDF + Web ===");


        // Routage intelligent : déterminer si la question concerne l'IA
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la question suivante porte sur " +
                        "l'intelligence artificielle ?\n" +
                        "Question : {{question}}\n" +
                        "Réponds seulement par oui, non ou peut-être."
        );

        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                String prompt = promptTemplate.apply(
                        Map.of("question", query.text())
                ).text();

                String decision = model.chat(prompt)
                        .trim()
                        .toLowerCase(Locale.ROOT)
                        .replaceAll("[.!?]+$", "")
                        .trim();

                System.out.println(
                        "\nDécision du routeur : " + decision
                );

                if (decision.equals("non")) {
                    System.out.println("Source utilisée : WEB uniquement");
                    return List.of(webRetriever);
                }

                System.out.println("Sources utilisées : PDF + WEB");
                return List.of(pdfRetriever, webRetriever);
            }
        };

        System.out.println("\n=== Routage intelligent PDF + Web ===");
        System.out.println("QueryRouter personnalisé créé avec succès !");


        // Création du RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println(
                "RetrievalAugmentor créé avec succès !"
        );

        // ==========================================
        // PHASE 6 : CRÉATION DE L'ASSISTANT
        // ==========================================

        // Mémoire de conversation : 10 messages
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // Assistant connecté au modèle Gemini
        // et au système RAG PDF + Web
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(chatMemory)
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        System.out.println(
                "Assistant RAG PDF + Web créé avec succès !"
        );

        // ==========================================
        // PHASE 7 : CONVERSATION INTERACTIVE
        // ==========================================

        Scanner scanner = new Scanner(System.in);

        System.out.println(
                "\n=== Conversation RAG PDF + Web ==="
        );

        System.out.println("Tapez 'fin' pour terminer.");

        while (true) {

            System.out.print("\nVotre question : ");

            String question = scanner.nextLine().trim();

            // Arrêt de la conversation
            if (question.equalsIgnoreCase("fin")) {
                System.out.println("Fin de la conversation.");
                break;
            }

            // Vérification des questions vides
            if (question.isBlank()) {
                System.out.println(
                        "Veuillez saisir une question."
                );
                continue;
            }

            // Envoi de la question à l'assistant
            String reponse = assistant.chat(question);

            System.out.println("\nRéponse de Gemini :");
            System.out.println(reponse);
        }

        scanner.close();
    }
}
