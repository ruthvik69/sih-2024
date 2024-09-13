import React from "react";
import { FlatList, Image, ScrollView, View } from "react-native";
import { Card, Text } from "react-native-paper";

interface ArchivedResult {
	id: string;
	name: string;
	disease: string;
	accuracy: number;
	date: Date;
	image: string;
	description?: string;
}

const arvhivedResults: ArchivedResult[] = [
	{
		id: "1",
		name: "Apple",
		disease: "Apple Scab",
		accuracy: 0.98,
		date: new Date("2024-09-01"),
		image: "https://raw.githubusercontent.com/sairamreddy77/sih/master/test/AppleCedarRust1.JPG",
		description:
			"Apple scab is a common disease of apple and crabapple trees, as well as mountain ash and pear. It is caused by the fungus Venturia inaequalis, which overwinters in infected leaves on the ground. In spring, spores are released and spread to newly emerging leaves and fruit. The disease is most severe in wet weather, and can cause defoliation and fruit loss. Fungicides can be used to control the disease, and resistant varieties are available.",
	},
	{
		id: "2",
		name: "Tomato",
		disease: "Blight",
		accuracy: 0.95,
		date: new Date("2024-09-04"),
		image: "https://raw.githubusercontent.com/sairamreddy77/sih/master/test/TomatoEarlyBlight1.JPG",
		description:
			"Early blight is a common disease of tomato plants, caused by the fungus Alternaria solani. It is characterized by dark brown spots on the leaves, which can enlarge and cause the leaves to turn yellow and die. The disease is most severe in warm, wet weather, and can cause defoliation and fruit loss. Fungicides can be used to control the disease, and resistant varieties are available.",
	},
	{
		id: "3",
		name: "Potato",
		disease: "Potato Late Blight",
		accuracy: 0.96,
		date: new Date("2024-09-09"),
		image: "https://raw.githubusercontent.com/sairamreddy77/sih/master/test/PotatoEarlyBlight1.JPG",
		description:
			"Late blight is a common disease of potato and tomato plants, caused by the fungus Phytophthora infestans. It is characterized by dark brown spots on the leaves, which can enlarge and cause the leaves to turn yellow and die. The disease is most severe in warm, wet weather, and can cause defoliation and fruit loss. Fungicides can be used to control the disease, and resistant varieties are available.",
	},
];

export default function Archives() {
	return (
		<View style={{ backgroundColor: "#222", padding: 20, flex: 1 }}>
			<Text style={{ marginVertical: 30 }} variant="displayLarge">
				Archives
			</Text>
			<FlatList
				style={{ flex: 1, padding: 30 }}
				data={arvhivedResults}
				renderItem={({ item }) => (
					<Card
						style={{
							backgroundColor: "#223",
							padding: 20,
							marginVertical: 10,
						}}
					>
						<Card.Cover source={{ uri: item.image }} />
						<Card.Content style={{ paddingVertical: 10 }}>
							<Text variant="titleMedium">{item.name}</Text>
							<Text variant="titleLarge">{item.disease}</Text>
							<View
								style={{
									flexDirection: "row",
									justifyContent: "space-between",
									marginVertical: 10,
								}}
							>
								<Text>{item.accuracy * 100}% Confidence</Text>
								<Text>{item.date.toDateString()}</Text>
							</View>
							<Text variant="bodyLarge">{item.description}</Text>
						</Card.Content>
					</Card>
				)}
				keyExtractor={(item) => item.id}
			/>
		</View>
	);
}
