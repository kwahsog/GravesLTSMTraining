package orgkwahsog;

public class DeepLearningLauncher {

    //refactoring example code to create seperate launcher.
    public static void main (String args[]) {
        System.out.println("Begin Example:");
        try {
            GravesLSTMCharModellingExample.run();
        } catch (Exception ex) {
            System.out.println("Running failed: " + ex.getMessage());
        }
    }
}
