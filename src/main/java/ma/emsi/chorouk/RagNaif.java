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
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import java.util.List;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        System.out.println("Démarrage du Test 1 - RAG naïf");

        // 1. Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("PDF chargé avec succès !");
        System.out.println(
                "Nombre de caractères : " + document.text().length()
        );

        // Affichage d'un extrait du document
        System.out.println(
                document.text().substring(
                        0,
                        Math.min(500, document.text().length())
                )
        );

        // 2. Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(
                300,
                30
        );

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // Afficher les deux premiers segments
        for (int i = 0; i < Math.min(2, segments.size()); i++) {
            System.out.println("\n===== Segment " + (i + 1) + " =====");
            System.out.println(segments.get(i).text());
        }

        // 3. Création du modèle d'embeddings
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        System.out.println("Modèle d'embeddings initialisé.");

        // 4. Création des embeddings pour tous les segments
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println(
                "Nombre d'embeddings créés : " + embeddings.size()
        );

        // Afficher la dimension du premier embedding
        if (!embeddings.isEmpty()) {
            System.out.println(
                    "Dimension du premier embedding : "
                            + embeddings.getFirst().dimension()
            );
        }
        // 5. Création du magasin d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

// Enregistrement des embeddings et de leurs segments associés
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés dans le magasin vectoriel.");
        System.out.println("Nombre de segments stockés : " + segments.size());
        // Phase 2 : récupération des passages pertinents

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetriever créé avec succès !");
        System.out.println("Nombre maximum de résultats : 2");
        System.out.println("Score minimum : 0.5");
        // Phase 2 : création du modèle Gemini

        String apiKey = System.getenv("GEMINI_API_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La variable GEMINI_API_KEY n'est pas configurée."
            );
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

// Mémoire de conversation : 10 messages
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

// Création de l'assistant avec le RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("Modèle Gemini initialisé.");
        System.out.println("Mémoire de 10 messages configurée.");
        System.out.println("Assistant RAG créé avec succès !");
        // Phase 2 : première question au LLM avec RAG

        // Conversation avec Gemini sans recompilation

        Scanner scanner = new Scanner(System.in);

        System.out.println("\n=== Conversation RAG avec Gemini ===");
        System.out.println("Tapez 'quit' pour terminer.");

        while (true) {

            System.out.print("\nVotre question : ");
            String question = scanner.nextLine();

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
}
