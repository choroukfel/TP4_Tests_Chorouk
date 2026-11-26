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

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        System.out.println("Démarrage du Test 2 - Transformation des questions");

        // 1. Chargement du PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("PDF chargé avec succès !");
        System.out.println("Nombre de caractères : " + document.text().length());

        // Affichage d'un extrait du document
        System.out.println(
                document.text().substring(
                        0,
                        Math.min(500, document.text().length())
                )
        );

        // 2. Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // Affichage des deux premiers segments
        for (int i = 0; i < Math.min(2, segments.size()); i++) {
            System.out.println("\n===== Segment " + (i + 1) + " =====");
            System.out.println(segments.get(i).text());
        }

        // 3. Initialisation du modèle d'embeddings
        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        System.out.println("Modèle d'embeddings initialisé.");

        // 4. Création des embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings créés : " + embeddings.size());

        if (!embeddings.isEmpty()) {
            System.out.println(
                    "Dimension du premier embedding : "
                            + embeddings.getFirst().dimension()
            );
        }

        // 5. Stockage des embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés dans le magasin vectoriel.");
        System.out.println("Nombre de segments stockés : " + segments.size());

        // 6. Création du ContentRetriever
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

        // 7. Configuration du modèle Gemini
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

        System.out.println("Modèle Gemini initialisé.");

        // 8. Mémoire de conversation : 10 messages
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        System.out.println("Mémoire de 10 messages configurée.");

        // 9. Test 2 : transformation des questions
        QueryTransformer queryTransformer =
                CompressingQueryTransformer.builder()
                        .chatModel(model)
                        .build();

        System.out.println("QueryTransformer créé avec succès !");

        // 10. Configuration du RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .contentRetriever(contentRetriever)
                        .build();

        System.out.println("RetrievalAugmentor créé avec succès !");

        // 11. Création de l'assistant avec transformation des questions
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG avec QueryTransformer créé avec succès !");

        // 12. Conversation interactive
        Scanner scanner = new Scanner(System.in);

        System.out.println("\n=== Conversation RAG avec transformation des questions ===");
        System.out.println("Tapez 'fin' pour terminer.");

        while (true) {

            System.out.print("\nVotre question : ");
            String question = scanner.nextLine();

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
