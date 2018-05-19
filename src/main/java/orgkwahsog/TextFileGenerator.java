package orgkwahsog;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.stream.Stream;

/** Class to combine all text files (including subdirectories) inside a directory into new output file.
 *
 */
public class TextFileGenerator {

    private String outputPath;
    private String directoryPath;

    public TextFileGenerator(String outputPath, String directoryPath) {
        this.outputPath = outputPath;
        this.directoryPath = directoryPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = outputPath;
    }

    public String getDirectoryPath() {
        return directoryPath;
    }

    public void setDirectoryPath(String directoryPath) {
        this.directoryPath = directoryPath;
    }

    /**
     * Generates the output text file.
     * @throws IOException in case reading/writing files occurs.
     */
    public void generateTextFiles() throws IOException {
        //create output file
        Path output = Paths.get(outputPath);

        if (Files.exists(output)) {
            Files.delete(output);
            Files.createFile(output);
        } else {
            Files.createFile(output);
        }
        //directory of input files to combine
        Path directory = Paths.get(directoryPath);

        Stream<Path> textFiles = Files.walk(directory)
                .filter(s -> s.toString().endsWith(".txt"))
                .map(Path::toAbsolutePath)
                .sorted();

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
