# resolve-app-jobs-for-grouping-tags
The input file called the "Filtered Phrases.txt" contains all of the tags phrases that are required to be grouped.
And the script runs and performs the whole grouping of tags task.
Version 4 involved direct implementation with the sentence transformers.
Version 5 involved the txtai library based testing using the same set of sentence transformers with some additional checking logic for edge cases.
(Used the same models through txtai library only to make the script ready for faster performance during production stage).
