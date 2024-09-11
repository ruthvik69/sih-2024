import { router } from "expo-router";
import React, { useState, useEffect } from "react";
import { View, Image, Dimensions } from "react-native";

const LoadingScreen = () => {
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		setTimeout(() => {
			setLoading(false);
		}, 2000); // navigate to HomeScreen after 2 seconds
	}, []);

	if (loading) {
		const { width, height } = Dimensions.get("window");

		return (
			<View
				style={{
					flex: 1,
					justifyContent: "center",
					alignItems: "center",
				}}
			>
				<Image
					source={require("../assets/loading-image.png")}
					style={{ width, height, resizeMode: "cover" }}
				/>
			</View>
		);
	} else {
		router.navigate("/home");
	}
};

export default LoadingScreen;
