import { Asset } from "expo-asset";
import { router } from "expo-router";
import React, { useState, useEffect } from "react";
import { View, Image, Dimensions, ImageSourcePropType } from "react-native";

const LoadingScreen = () => {
	const image = Asset.fromModule(
		require("@/assets/images/loading-screen.png")
	) as ImageSourcePropType;
	useEffect(() => {
		setTimeout(() => {
			router.navigate("/home");
		}, 2000); // navigate to HomeScreen after 2 seconds
	}, []);

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
				source={image}
				style={{ width, height, resizeMode: "cover" }}
			/>
		</View>
	);
};

export default LoadingScreen;
