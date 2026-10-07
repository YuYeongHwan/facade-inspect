import InspectionView from "@/components/inspection/InspectionView";

export default async function InspectionPage({ params }: PageProps<"/inspections/[id]">) {
  const { id } = await params;
  return <InspectionView id={decodeURIComponent(id)} />;
}
