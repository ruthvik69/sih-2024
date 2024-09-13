import { router } from "expo-router";
import * as React from "react";
import { Image, StyleSheet } from "react-native";
import { Appbar, FAB, Text, useTheme } from "react-native-paper";
import { View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

const BOTTOM_APPBAR_HEIGHT = 80;
const MEDIUM_FAB_HEIGHT = 56;

const Home = () => {
	const { bottom } = useSafeAreaInsets();
	const theme = useTheme();

	return (
		<>
			<Appbar
				style={[
					styles.bottom,
					{
						height: BOTTOM_APPBAR_HEIGHT + bottom,
						backgroundColor: theme.colors.elevation.level2,
					},
				]}
				safeAreaInsets={{ bottom }}
			>
				<Appbar.Action
					icon="archive"
					onPress={() => {
						router.navigate("/archives");
					}}
				/>

				<FAB
					mode="flat"
					size="medium"
					icon="camera"
					onPress={() => {
						router.navigate("/camera");
					}}
					style={[
						styles.fab,
						{ top: (BOTTOM_APPBAR_HEIGHT - MEDIUM_FAB_HEIGHT) / 2 },
					]}
				/>
			</Appbar>
			<View
				style={{
					backgroundColor: "#222",
					flex: 1,
					padding: 40,
					gap: 10,
				}}
			>
				<Text variant="titleMedium">Home</Text>
				<Text variant="displayMedium">Welcome Back</Text>
				<Text variant="bodyLarge">
					Use the camera icon below to capture the image of the crop
					to analyze.
				</Text>
				<Image
					source={{
						uri: "https://th.bing.com/th/id/OIP.bnZIbbc3U-KimQrufihwfgHaDO?rs=1&pid=ImgDetMain",
					}}
					style={{ width: "100%", height: 200, marginVertical: 30 }}
				/>
				<Image
					source={{
						uri: "https://th.bing.com/th/id/OIP.cI8MeqgFyHtowviaoFhXeQAAAA?rs=1&pid=ImgDetMain",
					}}
					style={{ width: "100%", height: 200, marginVertical: 30 }}
				/>
			</View>
		</>
	);
};

const styles = StyleSheet.create({
	top: {
		backgroundColor: "#026500",
		position: "absolute",
		left: 0,
		right: 0,
		top: 0,
	},
	bottom: {
		backgroundColor: "aquamarine",
		position: "absolute",
		left: 0,
		right: 0,
		bottom: 0,
		zIndex: 20,
	},
	fab: {
		position: "absolute",
		right: 16,
	},
});

export default Home;
