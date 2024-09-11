import { useResults } from "@/contexts/results";
import React from "react";
import { Image, View } from "react-native";
import { Divider, Text, useTheme } from "react-native-paper";

export default function Result() {
	const theme = useTheme();
	const { result } = useResults();
	return (
		<View
			style={{
				backgroundColor: theme.colors.background,
				flex: 1,
				padding: 10,
			}}
		>
			<Text variant="titleLarge">Results Page</Text>
			<Text>Here you will see the results of the detection process</Text>
			<Divider
				style={{ margin: 20, backgroundColor: theme.colors.secondary }}
			/>
			{result ? (
				<View style={{ gap: 10 }}>
					<Text variant="displaySmall">{result.label}</Text>
					<Text>{result.confidence}% confidence</Text>
					<Image
						source={{ uri: result.image?.uri }}
						width={20}
						height={30}
						style={{ width: 200, height: 300 }}
					/>
					<Text>{result.description}</Text>
				</View>
			) : (
				<Text>No result found</Text>
			)}
		</View>
	);
}
