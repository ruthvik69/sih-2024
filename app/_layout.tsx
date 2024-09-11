import { ResultsProvider } from "@/contexts/results";
import { router, Slot } from "expo-router";
import React from "react";
import { Appbar, PaperProvider, useTheme } from "react-native-paper";

export default function BaseLayout() {
	const theme = useTheme();

	return (
		<PaperProvider>
			<ResultsProvider>
				<Appbar.Header
					style={{ backgroundColor: theme.colors.primary }}
				>
					<Appbar.Action
						icon={"arrow-left"}
						onPress={() => {
							router.canGoBack()
								? router.back()
								: router.navigate("/");
						}}
					/>
					<Appbar.Content title="Crop Disease Detection" />
				</Appbar.Header>
				<Slot />
			</ResultsProvider>
		</PaperProvider>
	);
}
