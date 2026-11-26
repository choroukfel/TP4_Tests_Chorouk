package ma.emsi.chorouk;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {

        System.out.println("=== TP4 - Test 3 : Routage ===");

        // PHASE 1 : Chargement et ingestion des documents

        // Chargement du document sur l'intelligence artificielle
        Document documentIA = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        // Chargement du document sur les réseaux informatiques
        Document documentReseaux = ClassPathDocumentLoader.loadDocument(
                "reseaux.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("\n=== Documents chargés ===");
        System.out.println("Document IA : " + documentIA.text().length()
                + " caractères");
        System.out.println("Document Réseaux : " + documentReseaux.text().length()
                + " caractères");

        // Modèle d'embeddings partagé par les deux documents
        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        // Découpage en segments de 300 tokens,
        // avec 30 tokens de chevauchement
        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        // Création de deux magasins vectoriels indépendants
        System.out.println("\n=== Préparation des magasins vectoriels ===");

        System.out.println("Document IA :");
        EmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore(
                        documentIA,
                        splitter,
                        embeddingModel
                );

        System.out.println("Document Réseaux :");
        EmbeddingStore<TextSegment> storeReseaux =
                creerEmbeddingStore(
                        documentReseaux,
                        splitter,
                        embeddingModel
                );

        System.out.println("Les deux magasins vectoriels sont créés !");

        // PHASE 2 : Récupération et routage

        // Retriever pour le document IA
        ContentRetriever retrieverIA =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeIA)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // Retriever pour le document Réseaux
        ContentRetriever retrieverReseaux =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeReseaux)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("\n=== ContentRetrievers ===");
        System.out.println("Retriever IA créé avec succès !");
        System.out.println("Retriever Réseaux créé avec succès !");

        // Récupération sécurisée de la clé Gemini
        String apiKey = System.getenv("GEMINI_API_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable GEMINI_API_KEY n'est pas configurée."
            );
        }

        // Modèle Gemini utilisé pour le routage et les réponses
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        // Association des retrievers à la description
        // de leurs sources documentaires
        Map<ContentRetriever, String> retrieverToDescription =
                new LinkedHashMap<>();

        retrieverToDescription.put(
                retrieverIA,
                "Support de cours sur l'intelligence artificielle, "
                        + "le RAG, le fine-tuning et les modèles de langage."
        );

        retrieverToDescription.put(
                retrieverReseaux,
                "Cours sur les réseaux informatiques, TCP/IP, "
                        + "DNS, DHCP, les adresses IP, "
                        + "les routeurs et la sécurité réseau."
        );

        // Gemini sélectionne le ou les retrievers adaptés
        QueryRouter queryRouter =
                new LanguageModelQueryRouter(
                        model,
                        retrieverToDescription
                );

        System.out.println("\n=== Routage ===");
        System.out.println("Modèle Gemini initialisé !");
        System.out.println("LanguageModelQueryRouter créé avec succès !");

        // Intégration du routeur dans le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // Mémoire de conversation de 10 messages
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("RetrievalAugmentor créé avec succès !");
        System.out.println("Assistant avec routage créé avec succès !");

        // PHASE 3 : Conversation interactive

        Scanner scanner = new Scanner(System.in);

        System.out.println("\n=== Conversation avec routage ===");
        System.out.println("Tapez 'quit' pour terminer.");

        while (true) {

            System.out.print("\nVotre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("quit")) {
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

    // Méthode commune d'ingestion d'un document :
    // découpage, embeddings et stockage vectoriel
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            Document document,
            DocumentSplitter splitter,
            EmbeddingModel embeddingModel) {

        // Découpage du document
        List<TextSegment> segments = splitter.split(document);

        // Calcul des embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        // Magasin vectoriel en mémoire
        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        // Enregistrement des embeddings avec leurs segments
        store.addAll(embeddings, segments);

        System.out.println("Segments enregistrés : "
                + segments.size());

        return store;
    }
}
