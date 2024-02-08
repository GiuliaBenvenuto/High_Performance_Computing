#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// Function to read the content of a text file into a string
std::string readFileContent(const std::string &filePath) {
    std::ifstream file(filePath);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

// Function to calculate the Levenshtein distance between two strings
int levenshteinDistance(const std::string &s1, const std::string &s2) {
    int len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

    for (int i = 0; i <= len1; ++i) dp[i][0] = i;
    for (int j = 0; j <= len2; ++j) dp[0][j] = j;

    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({ dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost });
        }
    }

    return dp[len1][len2];
}

int main() {
    // Replace these with the actual file paths
    std::string filePath1 = "/seq.txt";
    std::string filePath2 = "/mandel_prova3.txt";

    std::string text1 = readFileContent(filePath1);
    std::string text2 = readFileContent(filePath2);

    int distance = levenshteinDistance(text1, text2);
    int maxLen = std::max(text1.length(), text2.length());
    double percentageDifference = (maxLen == 0) ? 0.0 : (double(distance) / maxLen) * 100;

    std::cout << "The texts are " << percentageDifference << "% different." << std::endl;

    return 0;
}