
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

import java.util.List;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.model.input.PromptTemplate;

import java.util.Collection;
import java.util.Map;

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import java.util.Scanner;

public class Test3 {

    public static void main(String[] args) {

        System.out.println("=== TP4 - Test 4 : Pas de RAG ===");

        // PHASE 1 : Chargement du document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document rag.pdf chargé avec succès !");
        System.out.println("Nombre de caractères : "
                + document.text().length());

        // Découpage du document
        DocumentSplitter splitter =
                DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : "
                + segments.size());

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

        System.out.println("Magasin vectoriel créé avec succès !");
        System.out.println("Embeddings enregistrés : "
                + embeddings.size());

        // PHASE 2 : Création du récupérateur
        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("\n=== ContentRetriever ===");
        System.out.println("ContentRetriever créé avec succès !");

        // Initialisation du modèle Gemini
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

        System.out.println("Modèle Gemini initialisé !");

        // Le modèle décide si la question nécessite un RAG
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête suivante porte sur " +
                        "l'intelligence artificielle ?\n" +
                        "Requête : {{question}}\n" +
                        "Réponds seulement par oui, non ou peut-être."
        );

        // Routeur personnalisé
        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                String prompt = promptTemplate.apply(
                        Map.of("question", query.text())
                ).text();

                // Interrogation directe du ChatModel
                String decision = model.chat(prompt)
                        .trim()
                        .toLowerCase(java.util.Locale.ROOT)
                        .replaceAll("[.!?]+$", "")
                        .trim();

                System.out.println("Décision du routeur : "
                        + decision);

                // Pas de RAG si la réponse est "non"
                if (decision.equals("non")) {
                    System.out.println("RAG désactivé.");
                    return List.of();
                }

                // RAG pour "oui" ou "peut-être"
                System.out.println("RAG activé.");
                return List.of(contentRetriever);
            }
        };

        System.out.println("QueryRouter personnalisé créé !");

        // Création du RetrievalAugmentor avec le routeur personnalisé
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // Mémoire de conversation
        ChatMemory chatMemory =
                MessageWindowChatMemory.withMaxMessages(10);

        // Création de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("RetrievalAugmentor créé avec succès !");
        System.out.println("Assistant créé avec succès !");

        // Conversation interactive
        Scanner scanner = new Scanner(System.in);

        System.out.println("\n=== Conversation avec ou sans RAG ===");
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
