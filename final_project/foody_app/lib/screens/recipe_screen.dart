import 'package:flutter/material.dart';

class RecipeScreen extends StatelessWidget {
  final Map<String, dynamic> recipeData;

  const RecipeScreen({super.key, required this.recipeData});

  @override
  Widget build(BuildContext context) {
    // Extract data with safety checks (in case Gemini misses a field)
    final String title = recipeData['title'] ?? 'Delicious Meal';
    final String time = recipeData['cooking_time'] ?? 'Unknown';
    final String calories = recipeData['calories'] ?? 'Unknown';
    final String servings = recipeData['servings'] ?? '2 Servs';
    final List<dynamic> ingredients = recipeData['ingredients_used'] ?? [];
    final List<dynamic> steps = recipeData['instructions'] ?? [];

    return Scaffold(
      backgroundColor: Colors.white,
      body: CustomScrollView(
        slivers: [
          // 1. The Big Header Image (Using a placeholder for now)
          SliverAppBar(
            expandedHeight: 250.0,
            floating: false,
            pinned: true,
            backgroundColor: Colors.teal,
            flexibleSpace: FlexibleSpaceBar(
              title: Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  shadows: [Shadow(color: Colors.black45, blurRadius: 10)],
                ),
              ),
              background: Image.network(
                'https://source.unsplash.com/800x600/?food,dinner', // Random food image
                fit: BoxFit.cover,
              ),
            ),
          ),

          // 2. The Content
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // A. AI Identified Tags
                  Wrap(
                    spacing: 8,
                    children: ingredients.map((ing) => Chip(
                      label: Text(ing.toString()),
                      backgroundColor: Colors.teal.shade50,
                      labelStyle: const TextStyle(color: Colors.teal),
                    )).toList(),
                  ),
                  
                  const SizedBox(height: 20),

                  // B. The Stats Row (Time, Cals, Servings)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildStatItem(Icons.timer, time),
                      _buildStatItem(Icons.local_fire_department, calories),
                      _buildStatItem(Icons.people, servings),
                    ],
                  ),

                  const SizedBox(height: 30),
                  const Divider(),
                  const SizedBox(height: 10),

                  // C. The Instructions Header
                  const Text(
                    "Instructions",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.teal,
                    ),
                  ),
                  const SizedBox(height: 15),

                  // D. The List of Steps
                  ListView.builder(
                    padding: EdgeInsets.zero,
                    shrinkWrap: true, // Vital for nesting inside ScrollView
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: steps.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        leading: CircleAvatar(
                          backgroundColor: Colors.teal,
                          child: Text(
                            "${index + 1}",
                            style: const TextStyle(color: Colors.white),
                          ),
                        ),
                        title: Text(
                          steps[index].toString(),
                          style: const TextStyle(fontSize: 16, height: 1.5),
                        ),
                        contentPadding: const EdgeInsets.symmetric(vertical: 8),
                      );
                    },
                  ),
                  
                  const SizedBox(height: 50),
                  
                  // E. The "Start Cooking" Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {},
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.teal,
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      child: const Text(
                        "Start Cooking Now",
                        style: TextStyle(fontSize: 18, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Helper widget for the Time/Cal/Serv pills
  Widget _buildStatItem(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          Icon(icon, color: Colors.teal, size: 20),
          const SizedBox(width: 8),
          Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}