


import java.util.HashMap;
import java.util.ArrayList;

/*
 * ArgumentParser: This is a simple utility class to parse command line arguments.
 */

public class ArgumentParser {
    public HashMap<Integer, String> index;
    public ArrayList<String> arguments;
    public HashMap<String, String> named_arguments;

    public ArgumentParser(String[] args) {
        System.out.println("----------------------------------------");
        System.out.println("Initializing ArgumentParser ...");
        // Initialize
        this.index = new HashMap<Integer, String>();
        this.arguments = new ArrayList<String>();
        this.named_arguments = new HashMap<String, String>();

        // Process arguments
        String arg;
        for (int i = 0; i < args.length; i++) {
            arg = args[i];
            if (arg.startsWith("--") || arg.startsWith("-")) {
                String[] split = arg.split("="); // Split on the first occurrence of "="
                // Raise an error if the key is already in the dictionary
                if (this.named_arguments.containsKey(split[0])) {
                    throw new IllegalArgumentException("Argument " + split[0] + " already exists.");
                }

                if (split.length == 1) {
                    this.named_arguments.put(split[0], null); // If there is no "="
                } else {
                    this.named_arguments.put(split[0], split[1]);
                }
                this.index.put(i, split[0]);
            } else {
                this.arguments.add(arg);
                this.index.put(i, arg);
            }
        }
        System.out.println("Done.");
        System.out.println("----------------------------------------");
    }

    public HashMap<Integer, String> GetIndex() { return this.index; }
    public ArrayList<String> GetArguments() { return this.arguments; }
    public HashMap<String, String> GetNamedArguments() { return this.named_arguments; }
}
