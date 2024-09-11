import { router } from "expo-router";
import * as React from "react";
import { StyleSheet } from "react-native";
import { Appbar, FAB, PaperProvider, useTheme } from "react-native-paper";
import { useSafeAreaInsets } from "react-native-safe-area-context";

const BOTTOM_APPBAR_HEIGHT = 80;
const MEDIUM_FAB_HEIGHT = 56;

const MyComponent = () => {
	const { bottom } = useSafeAreaInsets();
	const theme = useTheme();

	return (
		<>
			<Appbar.Header>
				<Appbar.Content title="Crop Disease Detection" />
			</Appbar.Header>
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
				<Appbar.Action icon="archive" onPress={() => {}} />

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
	},
	fab: {
		position: "absolute",
		right: 16,
	},
});

export default MyComponent;
