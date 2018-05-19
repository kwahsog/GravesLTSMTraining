package orgkwahsog;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.stream.Stream;

public class TextFileGenerator {

        public static void main(String[] args) throws IOException
        {
            //create output file
            Path output = Paths.get("E:\\Output\\true1.txt");

            if (Files.exists(output)) {
                Files.delete(output);
                Files.createFile(output);
            } else {
                Files.createFile(output);
            }
            //directory of input files to combine
            Path directory = Paths.get("E:\\TextFiles\\");

            Stream<Path> textFiles = Files.walk(directory)
                    .filter(s -> s.toString().endsWith(".txt"))
                    .map(Path::toAbsolutePath)
                    .sorted();

            Stream<Path> filesToProcess = Files.list(directory);

            // Iterate all files and write to output
            textFiles.forEach(path -> {
                Stream<String> lines = null;
                try {
                    lines = Files.lines(path);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                lines.forEach(line -> {
                    String lineToWrite = line + System.lineSeparator();
                    try {
                        Files.write(output, lineToWrite.getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
            });
        }
}
